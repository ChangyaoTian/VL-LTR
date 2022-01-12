import argparse
import datetime
from functools import partial
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.models import create_model
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler

from datasets import build_dataset
from losses import DistillationLoss, PretrainSentLoss, LabelSmoothingCrossEntropy
from samplers import RASampler, WeightedDistributedSampler
from engine import calc_class_acc, evaluate_LT, evaluate_pretrain, train_one_epoch, select_sent
from optim_factory import create_optimizer
from mixup import Mixup

import models
import utils
import collections
import os.path as osp
import warnings

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('VL-LTR training and evaluation script', add_help=False)
    parser.add_argument('--fp32-resume', action='store_true', default=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--config', required=True, type=str, help='config')
    parser.add_argument('--pretrained-bert', default=None, type=str)
    parser.add_argument('--txt-embed-path', type=str, default=None, help='config')
    parser.add_argument('--vis-backbone-path', type=str, default=None, help='config')
    parser.add_argument('--two-branch', action='store_true', help='two branch output')
    parser.set_defaults(two_branch=False)
    parser.add_argument('--debug', action='store_true', help='cls and img txt contrastive learning')
    parser.set_defaults(debug=False)

    # NLP parameters
    parser.add_argument('--desc-path', default='', type=str)
    parser.add_argument('--context-length', default=0, type=int, help='max length of text description')
    parser.add_argument('--sent-length', default=64, type=int, help='max number of selected sentences')
    parser.add_argument('--cls-token-length', default=1, type=int, help='the length of cls token')
    parser.add_argument('--loss-type', default='CE', type=str, help='loss type')
    parser.add_argument('--pretrain-cvlp', action='store_true', help='sentence-level pretraining')
    parser.set_defaults(pretrain_cvlp=False)
    parser.add_argument('--pretrain-cvlp-path', default='', type=str,
                        help='path of sentence-level pretraining task ckpt')

    # Model parameters
    parser.add_argument('--model', default='pvt_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--img-grad', action='store_true', default=True)
    parser.add_argument('--no-img-grad', action='store_false', dest='img_grad')
    parser.add_argument('--train-mode', action='store_true', default=True)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--text-lr', type=float, default=0, metavar='LR',
                        help='learning rate for text model (default: 0)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--clip-ms', action='store_true', help='use clip mean & std for initialization')
    parser.set_defaults(clip_ms=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default=None, type=str, metavar='MODEL',
                        help='Name of teacher model to train')
    parser.add_argument('--teacher-path', type=str, default=None)
    parser.add_argument('--distillation-type', default='none', choices=['none', 'feat', 'logits', 'logits_kl'],
                        type=str, help="")
    parser.add_argument('--distillation-alpha', default=0, type=float, help="")
    parser.add_argument('--distillation-beta', default=0, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    parser.add_argument('--distillation-training-mode', action='store_true', help="")
    parser.set_defaults(distillation_training_mode=False)

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--pretrained', action='store_true')

    # Sampler Parameters
    parser.add_argument('--weight-sample', action='store_true')
    parser.add_argument('--no-weight-sample', action='store_false', dest='weight_sample')
    parser.set_defaults(weight_sample=False)
    parser.add_argument('--use-sqrt-freq', action='store_true')
    parser.set_defaults(use_sqrt_freq=False)

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['PLACES_LT', 'CIFAR', 'IMNET', 
                                                                'IMNET_LT', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')

    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output-dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only', default=False)
    parser.add_argument('--test', action='store_true', help='Perform test only', default=False)
    parser.set_defaults(test=False)
    parser.add_argument('--test-p', action='store_true', help='Calculate acc for each class', default=False)
    parser.add_argument('--select', action='store_true', help='Perform test only', default=False)
    parser.set_defaults(select=False)
    parser.add_argument('--eval-pretrain', action='store_true', help='Perform evaluation for pretraining')
    parser.set_defaults(eval_pretrain=False)
    parser.add_argument('--ensemble', action='store_true', help='Perform zero-shot evaluation for pretraining like CLIP')
    parser.set_defaults(ensemble=False)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--drop-last', action='store_true')
    parser.add_argument('--no-drop-last', action='store_false', dest='drop_last')
    parser.set_defaults(drop_last=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communication during distributed "
                             "training")
    return parser


def main(args):
    utils.init_distributed_mode(args)
    # args.test = False
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(split="train", args=args)
    if args.test:
        dataset_test, _ = build_dataset(split="test", args=args)
    dataset_val, _ = build_dataset(split="val", args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        elif args.weight_sample:
            training_labels = np.array(dataset_train.targets).astype(int)
            train_class_counts = [len(training_labels[training_labels == l]) for l in range(args.nb_classes)]
            weights = 1. / torch.tensor(train_class_counts, dtype=torch.float)
            if args.use_sqrt_freq: weights.sqrt_()
            samples_weights = weights[list(dataset_train.targets)]
            sampler_train = WeightedDistributedSampler(
                dataset=dataset_train, weights=samples_weights, replacement=True,
                num_replicas=num_tasks, rank=global_rank, deterministic=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train,
                num_replicas=num_tasks,
                # num_replicas=0,
                rank=global_rank, shuffle=True,
                drop_last=args.drop_last
            )
            # sampler_train = torch.utils.data.RandomSampler(dataset_train)
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                # num_replicas=num_tasks,
                num_replicas=0,
                rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=args.drop_last,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    if args.test:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=torch.utils.data.SequentialSampler(dataset_test),
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes if not args.pretrain_cvlp else args.batch_size * utils.get_world_size()
        )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        dataset=dataset_train,
        args=args
    )

    model.to(device)

    model_ema = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    assert args.loss_type in ["softCE", "smoothCE", "BCE", "CE"]
    if args.mixup > 0. or args.loss_type == "softCE":
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        assert args.loss_type == "smoothCE"
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    elif args.loss_type == "BCE":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        assert args.loss_type == "CE"
        criterion = torch.nn.CrossEntropyLoss()
    print("using loss: ", str(criterion))
    if args.pretrain_cvlp:
        criterion = PretrainSentLoss(
            criterion, loss_type=args.loss_type, args=args,
            alpha=args.distillation_alpha, beta=args.distillation_beta,
            distill_type=args.distillation_type, tau=args.distillation_tau,
            set_training_mode=args.distillation_training_mode
        )
    else:
        criterion = DistillationLoss(
            criterion, None, 'none', 0, 0
        )
    print("using loss: ", str(criterion.__class__))

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.endswith('RN50.pt'):
            checkpoint = {}
            checkpoint['model'] = torch.jit.load(args.resume, map_location='cpu').state_dict()
        elif args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        elif osp.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
        else:
            checkpoint = None

        if checkpoint is not None:
            if 'model' in checkpoint:
                msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            else:
                msg = model_without_ddp.load_state_dict(checkpoint, strict=False)
            print(msg)
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint:
                    try:
                        loss_scaler.load_state_dict(checkpoint['scaler'])
                    except:
                        pass

    if args.eval:
        anot = ""
        if args.resume and osp.exists(args.resume) and checkpoint is not None:
            task_name = args.resume.split("/")[-2]
            anot = "# epoch={}, task={}".format(checkpoint['epoch'], task_name)

        data_loader = data_loader_val
        prefix = 'val'
        if args.test and not args.select:
            data_loader = data_loader_test
            prefix = 'test'
        if args.select and not args.test:
            data_loader = data_loader_train
            prefix = 'train'
        print("eval dataset:", prefix)
        if args.test_p:
            class_test_stats = calc_class_acc(data_loader, model, device,
                                              args=args, tokens=None)
            if args.output_dir and utils.is_main_process():
                with (output_dir / ("%s_%s_class.txt" % (args.data_set, prefix))).open("a") as f:
                    f.write(json.dumps(class_test_stats) + "\n")
            return
        if args.select:
            eval_func = partial(select_sent, args=args)
        elif args.eval_pretrain:
            eval_func = partial(evaluate_pretrain, args=args, labels=dataset_train.targets)
        else:
            eval_func = partial(evaluate_LT, args=args,
                                tokens=None, labels=dataset_train.targets)
        test_stats = eval_func(data_loader, model, device, prefix=prefix)
        log_stats = {f'{prefix}_{k}': v for k, v in test_stats.items()}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(anot + "\n")
                f.write(json.dumps(log_stats) + "\n")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        loss_scaler._scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32_resume)

        if not args.debug:
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                args=args
            )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint_{epoch + 1}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = {}

        if args.pretrain_cvlp:
            if args.eval_pretrain:
                test_stats = evaluate_pretrain(data_loader_val, model, device, args=args, 
                                        load_cache=False, labels=dataset_train.targets)
        else:
            test_stats = evaluate_LT(data_loader_val, model, device, args=args,
                                     tokens=None, labels=dataset_train.targets)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    if args.test:
        if args.pretrain_cvlp:
            test_stats = evaluate_pretrain(data_loader_test, model, device, args=args, 
                                    load_cache=False, labels=dataset_train.targets, prefix='test')
        else:
            test_stats = evaluate_LT(data_loader_test, model, device, args=args,
                                     tokens=None, labels=dataset_train.targets, prefix='test')
        log_stats = {f'test_{k}': v for k, v in test_stats.items()}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VL-LTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args = utils.update_from_config(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
