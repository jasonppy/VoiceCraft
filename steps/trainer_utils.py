
import torch
import math
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
import copy
import numpy as np
from typing import List
from scipy.stats import lognorm
import logging

class StatefulDistributedSampler(Sampler[int]):
    def __init__(self, dataset, batch_size, num_replicas = None, rank = None, shuffle = True, seed = 0, drop_last = False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.cur_epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.continue_flag = False
    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        self.indices = indices

        if self.continue_flag:
            self.indices = self.indices[int(self.cur_step*self.batch_size):]
            self.num_samples = len(self.indices)
            self.continue_flag = False
                
    def __iter__(self):
        for idx in self.indices:
            yield idx  

    def set_epoch_resume(self, epoch, cur_step):
        self.epoch = epoch
        self.cur_step = cur_step
        self.continue_flag = True


class StatefulSampler(Sampler):
    def __init__(self, data_source_length, batch_size, use_random=True, seed=1, epoch=0):
        self.use_random = use_random
        self.data_source_length = data_source_length
        self.num_samples = self.data_source_length
        self.batch_size = batch_size
        self.continue_flag = False
        self.seed = seed
        self.epoch = epoch
        self.cur_step = 0

    def __len__(self):
        return self.num_samples

    def __iter__(self):

        for idx in self.indices:
            yield idx

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.use_random:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.indices = torch.randperm(self.data_source_length, generator=g).tolist()  # type: ignore[arg-type]
        else:
            self.indices = list(range(self.data_source_length))  # type: ignore[arg-type]
        if self.continue_flag == True:
            self.continue_flag = False
            self.indices = self.indices[int(self.cur_step*self.batch_size):]
        
        self.num_samples = len(self.indices)
    
    def set_epoch_resume(self, epoch, cur_step):
        self.epoch = epoch
        self.cur_step = cur_step
        self.continue_flag = True


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def print_model_info(model, print_model = False, print_params = True):
    if print_model:
        logging.info(model)
    if print_params:
        all_params = {}
        for name, p in model.named_parameters():
            name = name.split(".")[0]
            if name in all_params:
                all_params[name] += p.numel()
            else:
                all_params[name] = p.numel()
        logging.info("num of parameters of each components:")
        for name in all_params:
            logging.info(f"{name}: {all_params[name]/1000000.:.2f}m")


class DistributedDynamicBatchSampler(Sampler):
    """
    modified from SpeechBrian, https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/dataio/sampler.py#L307
    This BatchSampler batches examples together by grouping them by their length.

    Every example in the batch have approximately the same length and
    thus padding is minimized.
    This enables faster training on datasets
    where length of examples can vary significantly (e.g Librispeech).
    Inspired by: https://www.tensorflow.org/api_docs/python/tf/data/experimental/bucket_by_sequence_length

    Dynamic batching is performed by specifying a max_batch_length which is the
    upper limit for the sum of the length of examples in a batch:
    e.g., if ex1 has length 4, ex2 length 5 and if max_batch_length is set to 6
    ex1 and ex2 will be placed, alone, in two distinct batches.

    Length for each example can be obtained in two manners.
    If the input dataset is a DynamicItemDataset it can be obtained by specifying a
    length_func. Default assumes a "duration" entry is in the annotation.
    Length for each example can also be passed to this class upon instantiation
    by specifying a list containing the length for each example and passing it to
    lengths_list.

    Examples are grouped together by defining a set of possible discrete intervals
    (buckets). Examples whose length fall into these intervals can be batched together.

    The number of buckets can be specified by using the arg num_buckets.
    There is usually an optimal range for the value of this argument.

    If num_buckets == 1, all examples can be batched together. You have maximum randomization
    but your training speed will be slower due to the fact that a large amount of the values will be padding
    as long and short examples can be batched together.
    As the number of buckets grows only examples with similar
    length can be grouped together.
    This trades-off speed with randomization.
    TLDR: Low number -> better randomization, High number -> faster training.
    NOTE THAT: if set too high the training speed will decrease. If num_buckets -> number of examples in the dataset the batch size
    will be small impacting training speed and possibly performance.

    The buckets can also be specified by passing a list to the bucket_boundaries
    argument instead of specifying a left_bucket_length and a bucket_length_multiplier.

    Example
    -------
    >>> import torch
    >>> import speechbrain as sb
    >>> from speechbrain.dataio.sampler import DynamicBatchSampler
    >>> from speechbrain.dataio.dataset import DynamicItemDataset
    >>> from speechbrain.dataio.dataloader import SaveableDataLoader
    >>> from speechbrain.dataio.batch import PaddedBatch
    >>> import numpy as np
    >>> item_lengths = sorted([np.random.randint(10, 100) for x in range(20)])
    >>> dataset = {"ex_{}".format(x) : {"wav" :torch.randn(x)} for x in item_lengths}
    >>> dataset = DynamicItemDataset(dataset)
    >>> dataset.set_output_keys(["wav"])
    >>> length_func = lambda x : len(x) # trivial in this example
    >>> bsampler = DynamicBatchSampler(dataset, 20, 4, length_func, shuffle=False, batch_ordering='descending')
    >>> dataloader = SaveableDataLoader(dataset, batch_sampler=bsampler, collate_fn=PaddedBatch)
    >>> for i, b in enumerate(dataloader):
    ...     data, length = b["wav"]
    >>> assert data.shape[-1] == max(item_lengths)

    Arguments
    ---------
    dataset : torch.utils.data.Dataset
        Pytorch Dataset from which elements will be sampled.
    max_batch_length : int
        Upper limit for the sum of the length of examples in a batch.
        Should be chosen based on your GPU memory.
    num_buckets : int
        Number of discrete buckets used to group examples together.
        If num_buckets == 1, all examples can be batched together. As the number of buckets grows only examples with similar
        length can be grouped together. This trades-off speed with randomization.
        Low number -> better randomization, High number -> faster training.
        However if set too high the training speed will decrease. If num_buckets -> number of examples in the dataset the batch size
        will be small impacting training speed and possibly performance.
        NOTE: you have either to specify manually the bucket_boundaries or the number of buckets.
    length_func : callable
        Function used to get length of each example from the dataset.
        This argument can be used only when the dataset is a Speechbrain DynamicItemDataset object.
        Can be anything: e.g. lambda x: x["duration"]*16000 returns number of samples
        if duration key in the annotation is in seconds and the file has 16kHz sampling freq.
    shuffle : bool
        Whether or not shuffle examples between each epoch.
    batch_ordering : string
        If ``random``, batches are randomly permuted; otherwise ``ascending`` or ``descending`` sorted by length.
    max_batch_ex: int
        If set, it limits the maximum number of examples that can be in a batch superseeding max_batch_length
        in instances where the amount of examples will exceeed the value specified here.
        E.g. you have a lot of short examples and the batch size for those will be too high, you can use this argument
        to limit the batch size for these short examples.
    bucket_boundaries : list
        Overrides bucket_length_multiplier and left_bucket_length by specifying manually
        the buckets right boundaries.
    lengths_list: list
        Overrides length_func by passing a list containing the length of each example
        in the dataset. This argument must be set when the dataset is a plain
        Pytorch Dataset object and not a DynamicItemDataset object as length_func
        cannot be used on Pytorch Datasets.
    epoch : int
        The epoch to start at.
    drop_last : bool
         If ``True``, the sampler will drop the last examples which
         have not been grouped.
    verbose: bool
        If ``True``, log also the stats for each batch at the first epoch.
    """

    def __init__(
        self,
        dataset,
        args,
        num_replicas = None, 
        rank = None, 
        shuffle = True, 
        seed = 0, 
        drop_last = False,
        length_func=lambda x: x["duration"],
        batch_ordering: str = "random",
        max_batch_ex: int = None,
        bucket_boundaries: List[int] = [],
        lengths_list: List[int] = None,
        epoch: int = 0,
        verbose: bool = False,
    ):
        self.args = args
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.num_replicas = num_replicas
        self.rank = rank
        max_batch_length = self.args.max_num_tokens if dataset.split == "train" else self.args.val_max_num_tokens
        logging.info(f"max_num_tokens per GPU for {dataset.split} split: {max_batch_length}")
        num_buckets = self.args.num_buckets
        #############
        



        self._dataset = dataset
        self._ex_lengths = {}
        # ex_ids = self._dataset.data_ids
        self.verbose = verbose

        # We do not put a default on num_buckets to encourage users to play with this parameter
        if num_buckets is None and len(bucket_boundaries) == 0:
            raise RuntimeError(
                "Please specify either num_buckets or bucket boundaries."
                "Check the docs, and/or the tutorial !"
            )
        assert lengths_list != None
        max_len = int(self.args.audio_max_length * self.args.encodec_sr)
        lengths_list = [min(l, max_len) for l in lengths_list] # replace all utt whose length is longer than max_len to max_len, will also do this in __getitem__ in dataset
        for indx in range(len(lengths_list)):
            self._ex_lengths[str(indx)] = lengths_list[indx]
        # if lengths_list is not None:
        #     # take length of examples from this argument and bypass length_key
        #     for indx in range(len(lengths_list)):
        #         self._ex_lengths[str(indx)] = lengths_list[indx]
        # else:
        #     # use length func
        #     if not isinstance(dataset, DynamicItemDataset):
        #         raise NotImplementedError(
        #             "Dataset should be a Speechbrain DynamicItemDataset when using length function"
        #         )
        #     for indx in range(len(self._dataset)):
        #         self._ex_lengths[str(indx)] = length_func(
        #             self._dataset.data[ex_ids[indx]]
        #         )

        if len(bucket_boundaries) > 0:
            if not all([x >= 0 for x in bucket_boundaries]):
                raise ValueError(
                    "All elements in bucket boundaries should be non-negative (>= 0)."
                )
            if not len(set(bucket_boundaries)) == len(bucket_boundaries):
                raise ValueError(
                    "Bucket_boundaries should not contain duplicates."
                )
            np.testing.assert_array_equal(
                np.array(bucket_boundaries),
                np.array(sorted(bucket_boundaries)),
                err_msg="The arg bucket_boundaries should be an ascending sorted list of non negative values values!",
            )
            self._bucket_boundaries = np.array(sorted(bucket_boundaries))
        else:
            # use num_buckets
            self._bucket_boundaries = np.array(
                self._get_boundaries_through_warping(
                    # max_batch_length=max_batch_length,
                    max_batch_length=max(lengths_list),
                    num_quantiles=num_buckets,
                )
            )

        self._max_batch_length = max_batch_length
        self._shuffle_ex = shuffle
        self._batch_ordering = batch_ordering
        self._seed = seed
        self._drop_last = drop_last
        if max_batch_ex is None:
            max_batch_ex = np.inf
        self._max_batch_ex = max_batch_ex
        # Calculate bucket lengths - how often does one bucket boundary fit into max_batch_length?
        self._bucket_lens = [
            max(1, int(max_batch_length / self._bucket_boundaries[i]))
            for i in range(len(self._bucket_boundaries))
        ] + [1]
        self._epoch = epoch
        self._cur_step = 0
        self.continue_flag = False
        self._generate_batches()
        self.num_samples = int(math.floor(len(self._batches) / self.num_replicas))
        self.total_size = int(self.num_samples * self.num_replicas)
        self._replica_batches = self._batches[self.rank:self.total_size:self.num_replicas]
        assert len(self._replica_batches) == self.num_samples, f"len(self._batches): {len(self._batches)}, self.total_size: {self.total_size}, self.num_samples: {self.num_samples},len(self._replica_batches): {len(self._replica_batches)}"
        logging.info(f"len(self._batches): {len(self._batches)}")
        logging.info(f"self.num_replicas: {self.num_replicas}")
        logging.info(f"num of batches on each replica: {self.num_samples}")

    def get_durations(self, batch):
        """Gets durations of the elements in the batch."""
        return [self._ex_lengths[str(idx)] for idx in batch]

    def _get_boundaries_through_warping(
        self, max_batch_length: int, num_quantiles: int,
    ) -> List[int]:

        # NOTE: the following lines do not cover that there is only one example in the dataset
        # warp frames (duration) distribution of train data
        logging.info("Batch quantisation in latent space")
        # linspace set-up
        num_boundaries = num_quantiles + 1
        # create latent linearly equal spaced buckets
        latent_boundaries = np.linspace(
            1 / num_boundaries, num_quantiles / num_boundaries, num_quantiles,
        )
        # get quantiles using lognormal distribution
        quantiles = lognorm.ppf(latent_boundaries, 1)
        # scale up to to max_batch_length
        bucket_boundaries = quantiles * max_batch_length / quantiles[-1]
        # compute resulting bucket length multipliers
        length_multipliers = [
            bucket_boundaries[x + 1] / bucket_boundaries[x]
            for x in range(num_quantiles - 1)
        ]
        # logging
        logging.debug(
            "Latent bucket boundary - buckets: {} - length multipliers: {}".format(
                list(map("{:.2f}".format, bucket_boundaries)),
                list(map("{:.2f}".format, length_multipliers)),
            )
        )
        return list(sorted(bucket_boundaries))

    def _permute_batches(self):

        if self._batch_ordering == "random":
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch) # since the random seed is based on self._seed and self._epoch, it should be the same for different processes when using DDP, and therefore the generated order should be the same across different process, this is important, because each replica will only take a portion of it, we want to make sure they take a non-overlapping portion, and all of them constitute the entire dataset
            sampler = torch.randperm(
                len(self._batches), generator=g
            ).tolist()  # type: ignore
            tmp = []
            for idx in sampler:
                tmp.append(self._batches[idx])
            self._batches = tmp

        elif self._batch_ordering == "ascending":
            self._batches = sorted(
                self._batches,
                key=lambda x: max([self._ex_lengths[str(idx)] for idx in x]),
            )
        elif self._batch_ordering == "descending":
            self._batches = sorted(
                self._batches,
                key=lambda x: max([self._ex_lengths[str(idx)] for idx in x]),
                reverse=True,
            )
        else:
            raise NotImplementedError

    def _generate_batches(self):
        logging.info("DynamicBatchSampler: Generating dynamic batches")
        if self._shuffle_ex:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch) # since the random seed is based on self._seed and self._epoch, it should be the same for different processes when using DDP, and therefore the generated order should be the same across different process, this is important, because each replica will only take a portion of it, we want to make sure they take a non-overlapping portion, and all of them constitute the entire dataset
            sampler = torch.randperm(len(self._dataset), generator=g).tolist()  # type: ignore 
            # pyp note: this is actually randomly permoted indices
        else:
            # take examples as they are: e.g. they have been sorted
            sampler = range(len(self._dataset))  # type: ignore

        self._batches = []
        bucket_batches = [[] for i in self._bucket_lens]

        stats_tracker = [
            {"min": np.inf, "max": -np.inf, "tot": 0, "n_ex": 0}
            for i in self._bucket_lens
        ]

        for idx in sampler:
            # length of pre-sampled audio
            item_len = self._ex_lengths[str(idx)]
            # bucket to fill up most padding
            bucket_id = np.searchsorted(self._bucket_boundaries, item_len)
            # fill audio's duration into that bucket
            bucket_batches[bucket_id].append(idx)

            stats_tracker[bucket_id]["min"] = min(
                stats_tracker[bucket_id]["min"], item_len
            )
            stats_tracker[bucket_id]["max"] = max(
                stats_tracker[bucket_id]["max"], item_len
            )
            stats_tracker[bucket_id]["tot"] += item_len
            stats_tracker[bucket_id]["n_ex"] += 1
            # track #samples - why not duration/#frames; rounded up?
            # keep track of durations, if necessary

            if (
                len(bucket_batches[bucket_id]) >= self._bucket_lens[bucket_id]
                or len(bucket_batches[bucket_id]) >= self._max_batch_ex
            ):
                self._batches.append(bucket_batches[bucket_id])
                bucket_batches[bucket_id] = []
                # keep track of durations

        # Dump remaining batches
        if not self._drop_last:
            for batch in bucket_batches:
                if batch:
                    self._batches.append(batch)

        self._permute_batches()  # possibly reorder batches

        if self._epoch == 0:  # only log at first epoch
            # frames per batch & their padding remaining
            boundaries = [0] + self._bucket_boundaries.tolist()

            for bucket_indx in range(len(self._bucket_boundaries)):
                try:
                    num_batches = stats_tracker[bucket_indx]["tot"] // (
                        self._max_batch_length
                    )
                    pad_factor = (
                        stats_tracker[bucket_indx]["max"]
                        - stats_tracker[bucket_indx]["min"]
                    ) / (
                        stats_tracker[bucket_indx]["tot"]
                        / stats_tracker[bucket_indx]["n_ex"]
                    )
                except ZeroDivisionError:
                    num_batches = 0
                    pad_factor = 0

                logging.debug(
                    (
                        "DynamicBatchSampler: Bucket {} with boundary {:.1f}-{:.1f} and "
                        + "batch_size {}: Num Examples {:.1f}, Num Full Batches {:.3f}, Pad Factor {:.3f}."
                    ).format(
                        bucket_indx,
                        boundaries[bucket_indx],
                        boundaries[bucket_indx + 1],
                        self._bucket_lens[bucket_indx],
                        stats_tracker[bucket_indx]["n_ex"],
                        num_batches,
                        pad_factor * 100,
                    )
                )

            if self.verbose:
                batch_stats = {
                    "tot_frames": [],
                    "tot_pad_frames": [],
                    "pad_%": [],
                }
                for batch in self._batches:
                    tot_frames = sum(
                        [self._ex_lengths[str(idx)] for idx in batch]
                    )
                    batch_stats["tot_frames"].append(tot_frames)
                    max_frames = max(
                        [self._ex_lengths[str(idx)] for idx in batch]
                    )
                    tot_pad = sum(
                        [
                            max_frames - self._ex_lengths[str(idx)]
                            for idx in batch
                        ]
                    )
                    batch_stats["tot_pad_frames"].append(tot_pad)
                    batch_stats["pad_%"].append(tot_pad / tot_frames * 100)

                padding_details = "Batch {} with {:.1f} frames with {} files - {:.1f} padding, {:.2f} (%) of total."
                padding_details = "DynamicBatchSampler: " + padding_details
                for i in range(len(self._batches)):
                    logging.debug(
                        padding_details.format(
                            i,
                            batch_stats["tot_frames"][i],
                            len(self._batches[i]),
                            batch_stats["tot_pad_frames"][i],
                            batch_stats["pad_%"][i],
                        )
                    )

    def __iter__(self):

        for batch in self._replica_batches:
            yield batch


        # if self._shuffle_ex:  # re-generate examples if ex_ordering == "random"
        #     self._generate_batches()
        # if self._batch_ordering == "random":
        #     # we randomly permute the batches only --> faster
        #     self._permute_batches()

    def set_epoch(self, epoch):
        """
        You can also just access self.epoch, but we maintain this interface
        to mirror torch.utils.data.distributed.DistributedSampler
        """
        self._epoch = epoch
        self._generate_batches()
        self._replica_batches = self._batches[self.rank:self.total_size:self.num_replicas]
        self.num_samples = int(math.floor(len(self._batches) / self.num_replicas))
        assert len(self._replica_batches) == self.num_samples, f"len(self._batches): {len(self._batches)}, self.total_size: {self.total_size}, self.num_samples: {self.num_samples},len(self._replica_batches): {len(self._replica_batches)}"

        if self.continue_flag:
            self.continue_flag = False
            self._replica_batches = self._replica_batches[self._cur_step:]
            self.num_samples = len(self._replica_batches)
 

    def __len__(self):
        return self.num_samples
    
    def set_epoch_resume(self, epoch, cur_step):
        self.continue_flag = True
        self._epoch = epoch
        self._cur_step = cur_step
