import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
import tqdm
from pqmf import PQMF
import wandb
import commons
import utils
from data_utils import DistributedBucketSampler
from data_utils_speaker_tone_lang import TextAudioSpeakerToneLangLoader, TextAudioSpeakerToneLangCollate

from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
    DurationDiscriminator2,
    AVAILABLE_FLOW_TYPES,
    AVAILABLE_DURATION_DISCRIMINATOR_TYPES
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
    subband_stft_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
global_step = 0


# - base vits2 : Aug 29, 2023
def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6060'

    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    net_dur_disc = None
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

        # Initialize wandb
        wandb.init(
            project=hps.wandb_project if hasattr(hps, 'wandb_project') else "vits2-training",
            name=hps.model_dir.split('/')[-1],
            config={
                "learning_rate": hps.train.learning_rate,
                "epochs": hps.train.epochs,
                "batch_size": hps.train.batch_size,
                "n_speakers": hps.data.n_speakers,
                "n_tones": hps.data.n_tones,
                "n_languages": hps.data.n_languages,
                "segment_size": hps.train.segment_size,
                "filter_length": hps.data.filter_length,
                "hop_length": hps.data.hop_length,
                "win_length": hps.data.win_length,
                "mel_fmin": hps.data.mel_fmin,
                "mel_fmax": hps.data.mel_fmax,
            }
        )

    if os.name == 'nt':
        dist.init_process_group(backend='gloo', init_method='env://', world_size=n_gpus, rank=rank)
    else:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:  # P.incoder for vits2
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80  # vits2
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    train_dataset = TextAudioSpeakerToneLangLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)

    collate_fn = TextAudioSpeakerToneLangCollate()
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
                              collate_fn=collate_fn, batch_sampler=train_sampler)
    if rank == 0:
        eval_dataset = TextAudioSpeakerToneLangLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                 batch_size=hps.train.batch_size, pin_memory=True,
                                 drop_last=False, collate_fn=collate_fn)
    # some of these flags are not being used in the code and directly set in hps json file.
    # they are kept here for reference and prototyping.

    if "use_transformer_flows" in hps.model.keys() and hps.model.use_transformer_flows == True:
        use_transformer_flows = True
        transformer_flow_type = hps.model.transformer_flow_type
        print(f"Using transformer flows {transformer_flow_type} for VITS2")
        assert transformer_flow_type in AVAILABLE_FLOW_TYPES, f"transformer_flow_type must be one of {AVAILABLE_FLOW_TYPES}"
    else:
        print("Using normal flows for VITS1")
        use_transformer_flows = False

    if "use_spk_conditioned_encoder" in hps.model.keys() and hps.model.use_spk_conditioned_encoder == True:
        if hps.data.n_speakers == 0:
            print("Warning: use_spk_conditioned_encoder is True but n_speakers is 0")
        print("Setting use_spk_conditioned_encoder to False as model is a single speaker model")
        use_spk_conditioned_encoder = False
    else:
        print("Using normal encoder for VITS1 (cuz it's single speaker after all)")
        use_spk_conditioned_encoder = False

    if "use_noise_scaled_mas" in hps.model.keys() and hps.model.use_noise_scaled_mas == True:
        print("Using noise scaled MAS for VITS2")
        use_noise_scaled_mas = True
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        use_noise_scaled_mas = False
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0

    if "use_duration_discriminator" in hps.model.keys() and hps.model.use_duration_discriminator == True:
        # print("Using duration discriminator for VITS2")
        use_duration_discriminator = True

        #- for duration_discriminator2
        # duration_discriminator_type = getattr(hps.model, "duration_discriminator_type", "dur_disc_1")
        duration_discriminator_type = hps.model.duration_discriminator_type
        print(f"Using duration discriminator {duration_discriminator_type} for VITS2")
        assert duration_discriminator_type in AVAILABLE_DURATION_DISCRIMINATOR_TYPES.keys(), f"duration_discriminator_type must be one of {list(AVAILABLE_DURATION_DISCRIMINATOR_TYPES.keys())}"
        #DurationDiscriminator = AVAILABLE_DURATION_DISCRIMINATOR_TYPES[duration_discriminator_type]

        if duration_discriminator_type == "dur_disc_1":
            net_dur_disc = DurationDiscriminator(
                hps.model.hidden_channels,
                hps.model.hidden_channels,
                3,
                0.1,
                gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
            ).cuda(rank)
        elif duration_discriminator_type == "dur_disc_2":
            net_dur_disc = DurationDiscriminator2(
                hps.model.hidden_channels,
                hps.model.hidden_channels,
                3,
                0.1,
                gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
            ).cuda(rank)
        '''
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(rank)
        '''
    else:
        print("NOT using any duration discriminator like VITS1")
        net_dur_disc = None
        use_duration_discriminator = False

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps)
    else:
        optim_dur_disc = None

    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=False)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=False)

    if net_dur_disc is not None:  # 2의 경우
        net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=False)

    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                   optim_d)
        if net_dur_disc is not None:  # 2의 경우
            _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                                                       net_dur_disc, optim_dur_disc)
        # epoch_str is actually the iteration/global_step from the checkpoint
        global_step = epoch_str
        epoch_str = max(1, global_step // len(train_loader))
        if rank == 0:
            logger.info(f"Resuming from global_step: {global_step}, epoch: {epoch_str}")
    except:
        epoch_str = 1
        global_step = 0
        if rank == 0:
            logger.info("Starting training from scratch")

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    if net_dur_disc is not None:  # 2의 경우
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(optim_dur_disc, gamma=hps.train.lr_decay,
                                                                    last_epoch=epoch_str - 2)
    else:
        scheduler_dur_disc = None

    scaler = GradScaler("cuda", enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d, net_dur_disc], [optim_g, optim_d, optim_dur_disc],
                               [scheduler_g, scheduler_d, scheduler_dur_disc], scaler, [train_loader, eval_loader],
                               logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d, net_dur_disc], [optim_g, optim_d, optim_dur_disc],
                               [scheduler_g, scheduler_d, scheduler_dur_disc], scaler, [train_loader, None], None, None)
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()

    # Finish wandb run
    if rank == 0:
        wandb.finish()

    # Clean up distributed process group
    dist.destroy_process_group()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:  # vits2
        net_dur_disc.train()

    if rank == 0:
        loader = tqdm.tqdm(train_loader, desc='Loading training data')
    else:
        loader = train_loader

    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, sid, tid, lid) in enumerate(loader):
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = net_g.module.mas_noise_scale_initial - net_g.module.noise_scale_delta * global_step
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        sid, tid, lid = sid.cuda(non_blocking=True), tid.cuda(non_blocking=True), lid.cuda(non_blocking=True)

        with autocast("cuda", enabled=hps.train.fp16_run):
            y_hat, y_hat_mb, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (
                hidden_x, logw, logw_) = net_g(x, x_lengths, spec, spec_lengths, sid=sid, tid=tid, lid=lid)

            if hps.model.use_mel_posterior_encoder or hps.data.use_mel_posterior_encoder:
                mel = spec
            else:
                mel = spec_to_mel_torch(
                    #spec,
                    spec.float(),  # - for 16bit stability
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast("cuda", enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

            # Duration Discriminator
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x.detach(), x_mask.detach(), logw_.detach(),
                                                        logw.detach())  # logw is predicted duration, logw_ is real duration
                with autocast("cuda", enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                grad_norm_dur_disc = commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast("cuda", enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw)
            with autocast("cuda", enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)

                if hps.model.mb_istft_vits == True:
                    pqmf = PQMF(y.device)
                    y_mb = pqmf.analysis(y)
                    loss_subband = subband_stft_loss(hps, y_mb, y_hat_mb)
                else:
                    loss_subband = torch.tensor(0.0)

                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_subband
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']

                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl, loss_subband]

                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}

                if net_dur_disc is not None:  # 2인 경우
                    scalar_dict.update(
                        {"loss/dur_disc/total": loss_dur_disc_all, "grad_norm_dur_disc": grad_norm_dur_disc})
                scalar_dict.update(
                    {"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl,
                     "loss/g/subband": loss_subband})

                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

                # Log to wandb
                def get_scalar(value):
                    """Convert tensor or scalar to Python float"""
                    return value.item() if torch.is_tensor(value) else float(value)

                wandb_dict = {
                    "train/loss_gen_total": get_scalar(loss_gen_all),
                    "train/loss_disc_total": get_scalar(loss_disc_all),
                    "train/loss_gen": get_scalar(loss_gen),
                    "train/loss_disc": get_scalar(loss_disc),
                    "train/loss_fm": get_scalar(loss_fm),
                    "train/loss_mel": get_scalar(loss_mel),
                    "train/loss_dur": get_scalar(loss_dur),
                    "train/loss_kl": get_scalar(loss_kl),
                    "train/loss_subband": get_scalar(loss_subband),
                    "train/learning_rate": lr,
                    "train/grad_norm_d": grad_norm_d,
                    "train/grad_norm_g": grad_norm_g,
                    "train/epoch": epoch,
                }

                if net_dur_disc is not None:
                    wandb_dict.update({
                        "train/loss_dur_disc_total": get_scalar(loss_dur_disc_all),
                        "train/loss_dur_gen": get_scalar(loss_dur_gen),
                        "train/grad_norm_dur_disc": grad_norm_dur_disc,
                    })

                # Log individual generator and discriminator losses
                for i, v in enumerate(losses_gen):
                    wandb_dict[f"train/loss_gen_{i}"] = get_scalar(v)
                for i, v in enumerate(losses_disc_r):
                    wandb_dict[f"train/loss_disc_r_{i}"] = get_scalar(v)
                for i, v in enumerate(losses_disc_g):
                    wandb_dict[f"train/loss_disc_g_{i}"] = get_scalar(v)

                if net_dur_disc is not None:
                    for i, v in enumerate(losses_dur_disc_r):
                        wandb_dict[f"train/loss_dur_disc_r_{i}"] = get_scalar(v)
                    for i, v in enumerate(losses_dur_disc_g):
                        wandb_dict[f"train/loss_dur_disc_g_{i}"] = get_scalar(v)

                wandb.log(wandb_dict, step=global_step)

                # if net_dur_disc is not None: # - 보류?
                #   scalar_dict.update({"loss/dur_disc_r" : f"{losses_dur_disc_r}"})
                #   scalar_dict.update({"loss/dur_disc_g" : f"{losses_dur_disc_g}"})
                #   scalar_dict.update({"loss/dur_gen" : f"{loss_dur_gen}"})

                # image_dict = {
                #     "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                #     "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                #     "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                #     "all/attn": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy())
                # }
                # utils.summarize(
                #     writer=writer,
                #     global_step=global_step,
                #     images=image_dict,
                #     scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, global_step,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, global_step,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                if net_dur_disc is not None:
                    utils.save_checkpoint(net_dur_disc, optim_dur_disc, hps.train.learning_rate, global_step,
                                          os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)))

                prev_g = os.path.join(hps.model_dir, "G_{}.pth".format(global_step - 3 * hps.train.eval_interval))
                if os.path.exists(prev_g):
                    os.remove(prev_g)
                    prev_d = os.path.join(hps.model_dir, "D_{}.pth".format(global_step - 3 * hps.train.eval_interval))
                    if os.path.exists(prev_d):
                        os.remove(prev_d)
                        prev_dur = os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step - 3 * hps.train.eval_interval))
                        if os.path.exists(prev_dur):
                            os.remove(prev_dur)

        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()

    # Initialize metric accumulators
    total_mel_loss = 0.0
    total_kl_loss = 0.0
    total_dur_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, sid, tid, lid) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            sid, tid, lid = sid.cuda(0), tid.cuda(0), lid.cuda(0)

            # Forward pass through the model to compute losses
            y_hat, y_hat_mb, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), _ = generator(
                x, x_lengths, spec, spec_lengths, sid=sid, tid=tid, lid=lid
            )

            # Compute mel spectrogram
            if hps.model.use_mel_posterior_encoder or hps.data.use_mel_posterior_encoder:
                mel = spec
            else:
                mel = spec_to_mel_torch(
                    spec.float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax)

            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            # Calculate losses
            mel_loss = F.l1_loss(y_mel, y_hat_mel)
            kl_loss_val = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
            dur_loss = torch.sum(l_length.float())

            # Accumulate losses
            total_mel_loss += mel_loss.item()
            total_kl_loss += kl_loss_val.item()
            total_dur_loss += dur_loss.item()
            num_batches += 1


    # Compute average losses
    avg_mel_loss = total_mel_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_dur_loss = total_dur_loss / num_batches

    # Log validation metrics to wandb
    wandb_eval_dict = {
        "val/mel_loss": avg_mel_loss,
        "val/kl_loss": avg_kl_loss,
        "val/dur_loss": avg_dur_loss,
        "val/total_loss": avg_mel_loss * hps.train.c_mel + avg_kl_loss * hps.train.c_kl + avg_dur_loss,
    }
    wandb.log(wandb_eval_dict, step=global_step)

    generator.train()


if __name__ == "__main__":
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    main()
