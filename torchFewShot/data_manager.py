from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader
import numpy as np
import transforms as T
import datasets
import dataset_loader
import PIL.Image as Image
from torchvision.transforms import InterpolationMode

class DataManager(object):
    """
    Few shot data manager
    """

    def __init__(self, args, use_gpu, collapse=False):
        super(DataManager, self).__init__()
        self.args = args
        self.use_gpu = use_gpu
        self.collapse = collapse

        print("Initializing dataset {}".format(args.dataset))
        dataset = datasets.init_imgfewshot_dataset(name=args.dataset)

        if args.load:
            transform_train = T.Compose([
                T.RandomCrop(84, padding=8),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(0.5)
            ])

            transform_test = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            if args.dataset in ['CIFARFS']:
                mean, std = [0.5071, 0.4866, 0.4409], [0.2009, 0.1984, 0.2023]
            else:
                mean, std = np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])

            transform_train = T.Compose([
                T.RandomResizedCrop((84, 84), interpolation=InterpolationMode.LANCZOS),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])

            transform_test = T.Compose([
                T.Resize((92, 92), interpolation=InterpolationMode.LANCZOS),
                T.CenterCrop((84, 84)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])

        #pin_memory = True if use_gpu else False
        pin_memory = False
        self.trainloader = DataLoader(
                dataset_loader.init_loader(name='train_loader',
                    dataset=dataset.train,
                    labels2inds=dataset.train_labels2inds,
                    labelIds=dataset.train_labelIds,
                    nKnovel=args.nKnovel,
                    nExemplars=args.nExemplars,
                    nTestNovel=args.train_nTestNovel,
                    epoch_size=args.train_epoch_size,
                    transform=transform_train,
                    load=args.load,
                    tiered=args.tiered,   	
                ),
                batch_size=args.train_batch, shuffle=False, num_workers=args.workers,
                pin_memory=pin_memory, drop_last=True,
            )

        self.valloader = DataLoader(
                dataset_loader.init_loader(name='test_loader',
                    dataset=dataset.val,
                    labels2inds=dataset.val_labels2inds,
                    labelIds=dataset.val_labelIds,
                    nKnovel=args.nKnovel,
                    nExemplars=args.nExemplars,
                    nTestNovel=args.nTestNovel,
                    epoch_size=args.epoch_size,
                    transform=transform_test,
                    load=args.load,
                    tiered=args.tiered,
                ),
                batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
                pin_memory=pin_memory, drop_last=False,
        )
        self.testloader = DataLoader(
                dataset_loader.init_loader(name='test_loader',
                    dataset=dataset.test,
                    labels2inds=dataset.test_labels2inds,
                    labelIds=dataset.test_labelIds,
                    nKnovel=args.nKnovel,
                    nExemplars=args.nExemplars,
                    nTestNovel=args.nTestNovel,
                    epoch_size=args.epoch_size,
                    transform=transform_test,
                    load=args.load,
                    tiered=args.tiered,
                ),
                batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
                pin_memory=pin_memory, drop_last=False,
        )

    def return_dataloaders(self):
        if self.args.phase == 'test':
            return self.trainloader, self.testloader
        elif self.args.phase == 'val':
            return self.trainloader, self.valloader
