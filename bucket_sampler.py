import torch
import torch.utils.data.distributed


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.

    Batch size is scaled inversely with bucket length to keep memory usage uniform.
    The first batch of each epoch comes from the longest bucket for early OOM detection.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
        self.num_batches = sum(
            self.num_samples_per_bucket[i] // (self.num_replicas * self.batch_sizes[i])
            for i in range(len(self.buckets))
        )

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)

        # Compute per-bucket batch sizes (inversely proportional to bucket max length)
        shortest_max = self.boundaries[1]
        self.batch_sizes = []
        for i in range(len(buckets)):
            bucket_max = self.boundaries[i + 1]
            bucket_bs = max(1, int(self.batch_size * shortest_max / bucket_max))
            self.batch_sizes.append(bucket_bs)

        # Print bucket info
        print("Bucket sampler configuration:")
        for i in range(len(buckets)):
            lo = self.boundaries[i]
            hi = self.boundaries[i + 1]
            print(f"  Bucket {i}: length {lo}-{hi}, samples={len(buckets[i])}, batch_size={self.batch_sizes[i]}")

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_sizes[i]
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        longest_batch = None
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]
            bucket_bs = self.batch_sizes[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // bucket_bs):
                batch = [bucket[idx] for idx in ids_bucket[j*bucket_bs:(j+1)*bucket_bs]]
                # Save first batch from the longest bucket for early OOM detection
                if i == len(self.buckets) - 1 and longest_batch is None:
                    longest_batch = batch
                else:
                    batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]

        # Put longest batch first so OOM happens immediately, not hours in
        if longest_batch is not None:
            batches = [longest_batch] + batches

        self.batches = batches
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_batches
