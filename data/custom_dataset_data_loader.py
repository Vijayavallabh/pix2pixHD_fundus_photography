import torch.utils.data
import torch
from torch.utils.data.distributed import DistributedSampler
from data.base_data_loader import BaseDataLoader


def _configure_torch_multiprocessing():
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except (AttributeError, RuntimeError):
        pass


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        _configure_torch_multiprocessing()
        self.dataset = CreateDataset(opt)
        self.sampler = None
        shuffle = not opt.serial_batches
        if getattr(opt, 'is_distributed', False):
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=opt.world_size,
                rank=opt.rank,
                shuffle=shuffle,
            )
            shuffle = False
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=shuffle,
            sampler=self.sampler,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
