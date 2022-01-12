import os
import json
from re import split

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from mcloader import ClassificationDataset


CLIP_DEFAULT_MEAN = (0.4815, 0.4578, 0.4082)
CLIP_DEFAULT_STD  = (0.2686, 0.2613, 0.2758)


def build_dataset(split, args):
    assert split in ['train', 'test', 'val']
    is_train = split == "train"
    transform = build_transform(is_train, args)

    assert args.data_set in ['PLACES_LT', 'INAT', 'IMNET', 'IMNET_LT']
    if args.data_set == "INAT":
        nb_classes = 8142
    elif args.data_set == "PLACES_LT":
        nb_classes = 365
    else:
        nb_classes = 1000
    dataset = ClassificationDataset(
        args.data_set,
        split,
        nb_classes=nb_classes,
        desc_path=args.desc_path,
        context_length=args.context_length,
        pipeline=transform,
        select=args.select
    )
    nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    DEFAULT_MEAN = CLIP_DEFAULT_MEAN if args.clip_ms else IMAGENET_DEFAULT_MEAN
    DEFAULT_STD  = CLIP_DEFAULT_STD  if args.clip_ms else IMAGENET_DEFAULT_STD
    if is_train:
        if args.aa == "":
            print("no auto augment")
            # use simple transform when dataset is IMNET_LT
            transform = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)
            ])
            return transform

        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=DEFAULT_MEAN,
            std=DEFAULT_STD,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD))
    return transforms.Compose(t)
