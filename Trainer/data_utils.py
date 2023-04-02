import numpy as np

import torch
from torchvision import transforms
from torch.utils.data.dataloader import default_collate

import dist_utils
from dataset import CoordsDataset, getWebDatasetWrapper
from bg_transform import RandomBackgroundTransform


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    if isinstance(batch[0][1], list) or isinstance(batch[0][1], tuple):
        targets = []
        for i in range(len(batch[0][1])):
            targets.append(torch.tensor([sample[1][i] for sample in batch], dtype=torch.int64))
    else:
        targets = torch.tensor([sample[1] for sample in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    c = len(imgs[0].mode)
    tensor = torch.zeros((len(imgs), c, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


def fast_collate_arrays(batch):
    seqs = [seq[0] for seq in batch]
    if isinstance(batch[0][1], list) or isinstance(batch[0][1], tuple):
        targets = []
        for i in range(len(batch[0][1])):
            targets.append(torch.tensor([sample[1][i] for sample in batch], dtype=torch.int64))
    else:
        targets = torch.tensor([sample[1] for sample in batch], dtype=torch.int64)
    tensor = torch.stack(seqs, dim=0)
    return tensor, targets


class PrefetchedWrapper(object):
    def prefetched_loader(loader):
        raw_normalizer = torch.tensor([255.0]).cuda().view(1, 1, 1, 1)
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True).float()
                if isinstance(next_target, list) or isinstance(next_target, tuple):
                    next_target = [x.cuda(non_blocking=True) for x in next_target]
                else:
                    next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.div_(raw_normalizer)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, validation=False):
        self.dataloader = dataloader
        self.epoch = 0
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self.validation = validation  # Just to make sure that the images are not shuffled after every iteration in order to make sure the logging still works

    def __len__(self):
        return int(np.ceil(len(self.dataloader.sampler) / float(self.dataloader.batch_size)))

    def __iter__(self):
        if not self.validation:
            if (self.dataloader.sampler is not None and isinstance(self.dataloader.sampler,
                                                                   torch.utils.data.distributed.DistributedSampler)):
                self.dataloader.sampler.set_epoch(self.epoch)
            self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader)


def get_dataloaders(args):
    use_prefetching_wrapper = True
    train_transform = []
    val_transform = []
    
    if args.use_rand_bg_aug:
        train_transform = [RandomBackgroundTransform()]
        val_transform = [RandomBackgroundTransform()]
    if args.use_rotation_aug:
        train_transform = [transforms.RandomRotation(degrees=(0, 360))]
    
    train_transform = train_transform + [transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),]
    val_transform = val_transform + [transforms.Resize(256), transforms.CenterCrop(224),]
    if not use_prefetching_wrapper:
        train_transform.append(transforms.ToTensor())
        val_transform.append(transforms.ToTensor())
    print("Train transform list:", train_transform)
    print("val transform list:", val_transform)
    
    train_transform = transforms.Compose(train_transform)
    val_transform = transforms.Compose(val_transform)
    
    collate_fn = default_collate
    if args.dataset in ["paperclip_wds", "3d_models_wds"]:
        assert args.train_tar_file is not None and args.val_tar_file is not None
        if args.distributed:
            raise NotImplementedError("WebDataset support with distributed training is not verified")
        multirot_stride = args.multirot_stride
        load_models = "3d_models_wds" in args.dataset
        train_set = getWebDatasetWrapper(args.train_tar_file, is_train=True, transform=train_transform, load_models=load_models, multirot_stride=multirot_stride)
        test_set = getWebDatasetWrapper(args.val_tar_file, is_train=False, transform=val_transform, load_models=load_models, multirot_stride=multirot_stride)
    elif args.dataset in ["paperclip_coords", "paperclip_coords_array", "3d_models_coords", "3d_models_coords_array"]:
        return_images = "array" not in args.dataset
        if not return_images:
            # Transform should only convert the np array into a tensor
            train_transform = torch.from_numpy
            val_transform = torch.from_numpy
            collate_fn = fast_collate_arrays  # Special collate function for arrays
        train_set = CoordsDataset(args.data, split="train", transform=train_transform, train_stride=args.train_stride, training_views=args.training_views,
                                  num_classes=args.num_classes, return_images=return_images, multirot_stride=args.multirot_stride)
        test_set = CoordsDataset(args.data, split="test", transform=val_transform, num_classes=args.num_classes, return_images=return_images,
                                 multirot_stride=args.multirot_stride)
        use_prefetching_wrapper = return_images  # Array size is odd
    else:
        raise RuntimeError(f"Unknown dataset name: {args.dataset}")
    
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'collate_fn': fast_collate if use_prefetching_wrapper else collate_fn} if args.use_cuda else {'collate_fn': default_collate}
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if args.distributed else None
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False if args.distributed or args.use_wds else True, sampler=train_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, sampler=test_sampler, **kwargs)
    
    if args.use_cuda and use_prefetching_wrapper:
        dist_utils.dist_print("Wrapping the Dataloader into prefetching wrapper...")
        train_loader = PrefetchedWrapper(train_loader, validation=False)
        test_loader = PrefetchedWrapper(test_loader, validation=True)

    dist_utils.dist_print(f"Dataset: {args.dataset} | Dataset class: {train_set.__class__.__name__} | # train examples: {len(train_set)} | # test examples: {len(test_set)}")
    dist_utils.dist_print(f"Batch size: {args.batch_size} | Optimizer batch size: {args.optimizer_batch_size} | Batch size multiplier: {args.batch_size_multiplier}")
    return train_loader, test_loader


def get_dataloader(dataset, args, test_set=False, always_shuffle=None):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.distributed else None
    
    shuffle = False if test_set or args.distributed else True
    batch_size = args.test_batch_size if test_set else args.batch_size
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'collate_fn': fast_collate if args.use_prefetching else default_collate} if args.use_cuda else {'collate_fn': default_collate}
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, **kwargs)
    
    if args.use_cuda and args.use_prefetching:
        dist_utils.dist_print("Wrapping the Dataloader into prefetching wrapper...")
        loader = PrefetchedWrapper(loader, validation=test_set)

    return loader
