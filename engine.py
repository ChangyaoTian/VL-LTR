"""
Train and eval functions used in main.py
"""
import math
import sys
import os.path as osp
from typing import Iterable, List, Optional, Tuple
from tqdm import tqdm
import time
import datetime
from collections import Counter

import torch
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SequentialSampler, DistributedSampler, RandomSampler

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import numpy as np

from models.layers import GatherLayer
import torch.nn.functional as F


def labels2idxs(labels: torch.Tensor):
    targets = torch.stack(
        [labels[i] == labels for i in range(labels.shape[0])])
    return targets


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    args=None):
    set_training_mode = args.train_mode
    fp32 = args.fp32_resume
    pretrain_cvlp = args.pretrain_cvlp
    two_branch = args.two_branch

    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    text_tokens = getattr(data_loader.dataset, 'text_tokens', None)
    sent_idxs = getattr(data_loader.dataset, 'end_idxs', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if pretrain_cvlp:
            idxs = [np.random.randint(sent_idxs[t]) for t in targets]
            tokens = torch.stack([text_tokens[targets[i]][idxs[i]] for i in range(len(targets))])
            tokens = tokens.to(device, non_blocking=True)

            if dist.is_initialized():
                targets = torch.cat(GatherLayer.apply(targets.contiguous()), 0)
            targets = labels2idxs(targets)
            targets = targets.type_as(samples).to(device, non_blocking=True)

            if mixup_fn is not None:
                targets_o = targets
                if dist.is_initialized():
                    samples = torch.cat(GatherLayer.apply(samples.contiguous()), 0)
                samples, targets = mixup_fn(samples, targets)
                if dist.is_initialized():
                    gpu_idx = utils.get_rank()
                    gpu_num = utils.get_world_size()
                    samples = samples.view(gpu_num, -1, samples.shape[1], samples.shape[2], samples.shape[3])[gpu_idx]
            samples = (samples, tokens)
        elif mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=not fp32):
            outputs = model(samples)
            if two_branch:
                loss0 = criterion(samples, outputs[0], targets)
                loss1 = criterion(samples, outputs[1], targets)
                loss = loss0 + loss1
                metric_logger.update(loss0=loss0)
                metric_logger.update(loss1=loss1)
                metric_logger.update(loss=loss)
            elif pretrain_cvlp:
                loss, distill_loss = criterion(samples, outputs, targets)
                metric_logger.update(distill_loss=distill_loss)
            else:
                loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if pretrain_cvlp:
            if mixup_fn is not None:
                targets = targets_o
            img_acc1 = multi_label_acc1(output=outputs[0], target=targets)
            text_acc1 = multi_label_acc1(output=outputs[1], target=targets)
            batch_size = samples[0].shape[0] / utils.get_world_size()
            metric_logger.meters['img_acc1'].update(img_acc1.item(), n=batch_size)
            metric_logger.meters['text_acc1'].update(text_acc1.item(), n=batch_size)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None, tokens=None):
    two_branch = args.two_branch

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    texts = tokens.to(device, non_blocking=True)

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model((images, texts))
            if two_branch:
                batch_size = images.shape[0]
                loss0 = criterion(output[0], target)
                loss1 = criterion(output[1], target)
                acc0_1, acc0_5 = accuracy(output[0], target, topk=(1, 5))
                acc1_1, acc1_5 = accuracy(output[1], target, topk=(1, 5))

                metric_logger.update(loss0=loss0.item())
                metric_logger.update(loss1=loss1.item())
                metric_logger.meters['acc0_1'].update(acc0_1.item(), n=batch_size)
                metric_logger.meters['acc0_5'].update(acc0_5.item(), n=batch_size)
                metric_logger.meters['acc1_1'].update(acc1_1.item(), n=batch_size)
                metric_logger.meters['acc1_5'].update(acc1_5.item(), n=batch_size)
            else:
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                batch_size = images.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def shot_acc(preds, labels, train_class_count, many_shot_thr=100, low_shot_thr=20):
    # _, preds = output.topk(1, 1, True, True)
    # preds = preds.squeeze(-1)

    # [min_shot, max_shot, correct, total, acc]
    shot_cnt_stats = {
        "many": [many_shot_thr - 1, max(train_class_count), 0, 0, 0.],
        "median": [low_shot_thr, many_shot_thr - 1, 0, 0, 0.],
        "low": [0, low_shot_thr, 0, 0, 0.],
        "10-shot": [0, 10, 0, 0, 0.],
        "5-shot": [0, 5, 0, 0, 0.],
    }
    for l in torch.unique(labels):
        class_correct = torch.sum((preds[labels == l] == labels[labels == l])).item()
        test_class_count = len(labels[labels == l])
        for stat_name in shot_cnt_stats:
            stat_info = shot_cnt_stats[stat_name]
            if train_class_count[l] > stat_info[0] and train_class_count[l] <= stat_info[1]:
                stat_info[2] += class_correct
                stat_info[3] += test_class_count
    for stat_name in shot_cnt_stats:
        shot_cnt_stats[stat_name][-1] = shot_cnt_stats[stat_name][2] / shot_cnt_stats[stat_name][3] * \
                                        100.0 if shot_cnt_stats[stat_name][3] != 0 else 0.
    return shot_cnt_stats


@torch.no_grad()
def evaluate_LT(data_loader, model, device, args=None, tokens=None, labels=None, prefix='val'):
    two_branch = args.two_branch

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    texts = tokens.to(device, non_blocking=True) if tokens is not None else None

    training_labels = np.array(labels).astype(int)
    train_class_count = [len(training_labels[training_labels == l]) for l in range(args.nb_classes)]
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        inputs = (images, texts) if texts is not None else images
        # compute output
        with torch.cuda.amp.autocast():
            output = model(inputs)

            if two_branch:
                batch_size = images.shape[0]
                loss0 = criterion(output[0], target)
                loss1 = criterion(output[1], target)
                loss = loss0 + loss1
                acc0_1, acc0_5 = accuracy(output[0], target, topk=(1, 5))
                acc1_1, acc1_5 = accuracy(output[1], target, topk=(1, 5))
                alpha = 0.7 if 'INAT' in args.data_set else 0.2
                acc1, acc5 = accuracy(output[0].softmax(1) * alpha + output[1].softmax(1) * (1-alpha), target, topk=(1, 5))

                metric_logger.update(loss=loss.item())
                metric_logger.update(loss0=loss0.item())
                metric_logger.update(loss1=loss1.item())
                metric_logger.meters['acc0_1'].update(acc0_1.item(), n=batch_size)
                metric_logger.meters['acc0_5'].update(acc0_5.item(), n=batch_size)
                metric_logger.meters['acc1_1'].update(acc1_1.item(), n=batch_size)
                metric_logger.meters['acc1_5'].update(acc1_5.item(), n=batch_size)
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

                output_ = output[0] + output[1]
                _, preds = output_.topk(1, 1, True, True)
                preds = preds.squeeze(-1)
                shot_cnt_stats = shot_acc(preds, target, train_class_count)
                for stat_name in shot_cnt_stats:
                    metric_logger.meters[stat_name].update(shot_cnt_stats[stat_name][-1],
                                                           n=shot_cnt_stats[stat_name][-2])
            else:
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                batch_size = images.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

                _, preds = output.topk(1, 1, True, True)
                preds = preds.squeeze(-1)
                shot_cnt_stats = shot_acc(preds, target, train_class_count)
                for stat_name in shot_cnt_stats:
                    metric_logger.meters[stat_name].update(shot_cnt_stats[stat_name][-1],
                                                           n=shot_cnt_stats[stat_name][-2])

    if two_branch:
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} '
              'Acc0@1 {top01.global_avg:.3f} Acc0@5 {top05.global_avg:.3f} '
              'Acc1@1 {top11.global_avg:.3f} Acc1@5 {top15.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5,
                      top01=metric_logger.acc0_1, top05=metric_logger.acc0_5,
                      top11=metric_logger.acc1_1, top15=metric_logger.acc1_5,
                      losses=metric_logger.loss))
    else:
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def calc_class_acc(data_loader, model, device, args=None, tokens=None, prefix='val'):
    '''calculate accuracy for each class separately'''
    criterion = torch.nn.CrossEntropyLoss()
    two_branch = args.two_branch
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    texts = tokens.to(device, non_blocking=True) if tokens is not None else None

    labels = data_loader.dataset.targets
    labels = np.array(labels).astype(int)
    cnt_per_class = [len(labels[labels == l]) for l in range(args.nb_classes)]
    true_per_class = [0] * args.nb_classes
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        inputs = (images, texts) if texts is not None else images
        # compute output
        with torch.cuda.amp.autocast():
            output = model(inputs)
            if two_branch:
                loss0 = criterion(output[0], target)
                loss1 = criterion(output[1], target)
                loss = loss0 + loss1
                alpha = 0.7 if 'INAT' in args.data_set else 0.2
                acc1, acc5 = accuracy(output[0].softmax(1) * alpha + output[1].softmax(1) * (1-alpha), target, topk=(1, 5))
                output = output[0] + output[1]
            else:
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        _, preds = output.topk(1, 1, True, True)
        preds = preds.squeeze(-1)
        acc = preds == target
        for l in torch.unique(target):
            true_per_class[l] += torch.sum(acc[target == l]).item()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return [true_per_class[i] / cnt_per_class[i] for i in range(args.nb_classes)]


def multi_label_acc1(output: torch.Tensor, target: torch.Tensor):
    # target is a matrix of [0,1] with the same shape as output
    # print("multi_label_acc1:", target.shape)
    assert output.shape == target.shape
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t().squeeze()
    return (target[torch.arange(0, target.shape[0]), pred] == 1).sum(
        0) * 100. / target.shape[0]


@torch.no_grad()
def evaluate_pretrain(data_loader: DataLoader, model, device, labels=None, args=None, load_cache=True, topk=(5, 1),
                      prefix='val'):
    # switch to evaluation mode
    start_time = time.time()
    model.eval()
    if args.distributed: model = model.module

    text_tokens = getattr(data_loader.dataset, 'text_tokens', None)
    assert text_tokens is not None and isinstance(text_tokens, List), \
        "text_tokens is None, This function only supports pretraining phase"
    text_tokens = torch.cat(text_tokens)
    sent_idxs = getattr(data_loader.dataset, 'end_idxs', None)
    assert sent_idxs is not None and isinstance(sent_idxs, List)
    targets = torch.tensor(data_loader.dataset.targets).to(device)
    text_targets = torch.empty((sum(sent_idxs),), dtype=torch.long).to(device)  # [Nt,]
    left = 0
    for i in range(len(sent_idxs)):
        text_targets[left : left + sent_idxs[i]] = i
        left += sent_idxs[i]
    
    # step 1. obtain all embeddings of image and text
    image_embeddings, text_embeddings = None, None
    if args.resume:
        cache_dir = osp.dirname(args.resume)
        img_embed_path = osp.join(cache_dir, "%s_img_embed.npy" % prefix)
        txt_embed_path = osp.join(cache_dir, "txt_embed.npy")
    if load_cache and osp.exists(img_embed_path):
        print("using cached image embeddings")
        image_embeddings = torch.from_numpy(np.load(img_embed_path)).to(device, non_blocking=True)
    if load_cache and osp.exists(txt_embed_path):
        print("using cached text embeddings")
        text_embeddings = torch.from_numpy(np.load(txt_embed_path)).to(device, non_blocking=True)

    # image
    if image_embeddings is None:
        image_embeddings = []
        iter = tqdm(data_loader, desc="image embeddings") if load_cache else data_loader
        for images, target in iter:
            images = images.to(device, non_blocking=True)
            # compute output
            with torch.cuda.amp.autocast():
                image_features = model.encode_image(images)
            image_embeddings.append(image_features.detach())
        image_embeddings = torch.cat(image_embeddings)
        if utils.is_main_process(): np.save(img_embed_path, image_embeddings.cpu().numpy())
    # print("image_embeddings.shape: ", image_embeddings.shape) # [Ni, 1024]

    # text
    if text_embeddings is None:
        text_embeddings = []
        tokens_loader_val = DataLoader(
            text_tokens, sampler=SequentialSampler(text_tokens),
            batch_size=int(8 * args.batch_size),
            num_workers=args.num_workers, pin_memory=args.pin_mem,
            drop_last=False
        )
        iter = tqdm(tokens_loader_val, desc="text embeddings") if load_cache else tokens_loader_val
        for batch_tokens in iter:
            batch_tokens = batch_tokens.to(device, non_blocking=True)
            # compute output
            with torch.cuda.amp.autocast():
                text_features = model.encode_text(batch_tokens)
            text_embeddings.append(text_features.detach())
        text_embeddings = torch.cat(text_embeddings)
        if utils.is_main_process(): np.save(txt_embed_path, text_embeddings.cpu().numpy())
    # print("text_embeddings.shape: ", text_embeddings.shape) # [Nt, 1024]
    if args.ensemble:
        print("using ensemble")
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        n_text_embeddings = []
        left = 0
        for i in range(len(sent_idxs)):
            n_text_embeddings.append(torch.mean(text_embeddings[left : left + sent_idxs[i], :], dim=0))
            left += sent_idxs[i]
        text_embeddings = torch.stack(n_text_embeddings)
        text_targets = torch.arange(len(sent_idxs)).to(device)

    # step 2. compute cosine similarity for image and text
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    # image
    def get_pred(embeddings_A, embeddings_B, topk=1, desc=''):
        embeddings_loader = DataLoader(
            embeddings_A.cpu(), sampler=SequentialSampler(embeddings_A),
            batch_size=int(8 * args.batch_size),
            num_workers=args.num_workers, pin_memory=args.pin_mem,
            drop_last=False
        )
        iter = tqdm(embeddings_loader, desc=desc) if load_cache else embeddings_loader
        preds = []
        for batch_embeddings in iter:
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)
            batch_logits = batch_embeddings @ embeddings_B.t()
            _, batch_preds = batch_logits.topk(topk, dim=1, largest=True, sorted=True)  # [BN, topk]
            preds.append(batch_preds)
        preds = torch.cat(preds)
        return preds
    
    pred_image = get_pred(image_embeddings, text_embeddings, 
                            topk=max(topk), desc="preds of image embeddings") # [Ni, topk]
    print("pred_image.shape:", pred_image.shape)
    # print("logits_per_image.shape", logits_per_image.shape)

    pred_label = text_targets[pred_image]  # [Ni, topk]
    image_acc1 = torch.sum(pred_label[:, 0] == targets) * 100.0 / pred_image.shape[0]
    # shot acc
    img_shot_acc, knn_shot_acc = {}, {}
    if labels is not None:
        training_labels = np.array(labels).astype(int)
        train_class_count = [len(training_labels[training_labels == l]) for l in range(args.nb_classes)]
        img_shot_acc = shot_acc(pred_label[:, 0], targets, train_class_count=train_class_count)
        img_shot_acc = {k: v[-1] for k, v in img_shot_acc.items()}
    # knn
    vote_result = torch.tensor([Counter(label.tolist()).most_common(1)[0][0] for label in pred_label]).to(device)
    if labels is not None:
        knn_shot_acc = shot_acc(vote_result, targets, train_class_count=train_class_count)
        knn_shot_acc = {f"knn_{k}": v[-1] for k, v in knn_shot_acc.items()}
    knn_acc = torch.sum(vote_result == targets) * 100.0 / pred_image.shape[0]
    pred_text = get_pred(text_embeddings, image_embeddings, topk=1, desc="preds of text embeddings")
    pred_text = pred_text.squeeze() # [Nt, ]
    print("pred_text.shape:", pred_text.shape)
    pred_text = targets[pred_text]
    text_acc1 = torch.sum(pred_text == text_targets) * 100.0 / pred_text.shape[0]

    torch.cuda.synchronize()
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print("* image_Acc@1: {:.3f}% text_Acc@1 {:.3f}% knn_Acc@5 {:.3f}% Total time: {}".format(
        image_acc1, text_acc1, knn_acc, total_time))
    return {"image_acc1": image_acc1.item(), "text_acc1": text_acc1.item(),
            f"knn_{max(topk)}": knn_acc.item(), **img_shot_acc, **knn_shot_acc}


@torch.no_grad()
def select_sent(data_loader: DataLoader, model, device, args=None, load_cache=True, topk=(5, 1), prefix='val'):
    # switch to evaluation mode
    start_time = time.time()
    model.eval()
    if args.distributed: model = model.module

    text_tokens = getattr(data_loader.dataset, 'text_tokens', None)
    assert text_tokens is not None and isinstance(text_tokens, List), \
        "text_tokens is None, This function only supports pretraining phase"
    text_tokens = torch.cat(text_tokens)
    sent_idxs = getattr(data_loader.dataset, 'end_idxs', None)
    assert sent_idxs is not None and isinstance(sent_idxs, List)
    text_targets = torch.empty((sum(sent_idxs),), dtype=torch.long).to(device)  # [Nt,]
    left = 0
    for i in range(len(sent_idxs)):
        text_targets[left : left + sent_idxs[i]] = i
        left += sent_idxs[i]

    # step 1. obtain all embeddings of image and text
    image_embeddings, text_embeddings, image_targets = None, None, None
    if args.resume:
        cache_dir = osp.dirname(args.resume)
        img_embed_path = osp.join(cache_dir, "%s_img_embed.npy" % prefix)
        img_target_path = osp.join(cache_dir, "%s_img_target.npy" % prefix)
        txt_embed_path = osp.join(cache_dir, "%s_txt_embed.npy" % prefix)
    if load_cache and osp.exists(img_embed_path):
        print("using cached image embeddings")
        image_embeddings = torch.from_numpy(np.load(img_embed_path)).to(device, non_blocking=True)
    if load_cache and osp.exists(img_target_path):
        print("using cached image targets")
        image_targets = torch.from_numpy(np.load(img_target_path)).to(device, non_blocking=True)
    if load_cache and osp.exists(txt_embed_path):
        print("using cached text embeddings")
        text_embeddings = torch.from_numpy(np.load(txt_embed_path)).to(device, non_blocking=True)

    # image
    if image_embeddings is None or image_targets is None:
        image_embeddings = []
        image_targets = []
        iter = tqdm(data_loader, desc="image embeddings") if load_cache else data_loader
        for images, target in iter:
            images = images.to(device, non_blocking=True)
            image_targets.append(target)
            # compute output
            with torch.cuda.amp.autocast():
                image_features = model.encode_image(images)
            image_embeddings.append(image_features.detach())
        image_embeddings = torch.cat(image_embeddings)
        image_targets = torch.cat(image_targets).to(device)
        if utils.is_main_process(): np.save(img_embed_path, image_embeddings.cpu().numpy())
        if utils.is_main_process(): np.save(img_target_path, image_targets.cpu().numpy())
    # print("image_embeddings.shape: ", image_embeddings.shape) # [Ni, 1024]

    # text
    if text_embeddings is None:
        text_embeddings = []
        tokens_loader_val = DataLoader(
            text_tokens, sampler=SequentialSampler(text_tokens),
            batch_size=int(8 * args.batch_size),
            num_workers=args.num_workers, pin_memory=args.pin_mem,
            drop_last=False
        )
        iter = tqdm(tokens_loader_val, desc="text embeddings") if load_cache else tokens_loader_val
        for batch_tokens in iter:
            batch_tokens = batch_tokens.to(device, non_blocking=True)
            # compute output
            with torch.cuda.amp.autocast():
                text_features = model.encode_text(batch_tokens)
            text_embeddings.append(text_features.detach())
        text_embeddings = torch.cat(text_embeddings)
        if utils.is_main_process(): np.save(txt_embed_path, text_embeddings.cpu().numpy())
    # print("text_embeddings.shape: ", text_embeddings.shape) # [Nt, 1024]

    # step 2. compute cosine similarity for image and text
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

    text_ces = []
    iter = tqdm(range(text_embeddings.shape[0]), desc="ce for text embeddings") if load_cache else range(text_embeddings.shape[0])
    for i in iter:
        text_embedding = text_embeddings[i]
        logit = text_embedding @ image_embeddings.t() * model.logit_scale.exp()
        label = image_targets == text_targets[i]
        label = label / label.sum()
        ce = torch.sum(-label * F.log_softmax(logit, dim=-1), dim=-1)
        text_ces.append(ce)

    text_ces = torch.tensor(text_ces).to(device)

    txt_ce_path = osp.join(cache_dir, "%s_txt_ce.npy" % prefix)
    if utils.is_main_process(): np.save(txt_ce_path, text_ces.cpu().numpy())
    exit(0)
