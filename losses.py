"""
Implements the knowledge distillation loss
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models import create_model

def cross_entropy(outputs, teacher_outputs):
    logprobs = F.log_softmax(outputs, dim=-1)
    soft_targets = F.softmax(teacher_outputs, dim=-1)
    distill_loss = -torch.sum(soft_targets * logprobs, dim=-1)
    return distill_loss.mean()


def kl_div(outputs1, outputs2, T=1.):
    return F.kl_div(
                F.log_softmax(outputs1 / T, dim=1),
                F.log_softmax(outputs2 / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs1.numel()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        logprobs = F.log_softmax(x, dim=-1)
        smooth_loss = -logprobs.mean(dim=-1)
        if target.dim() == 1:
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
        else:
            assert target.dim() == 2
            nll_loss = -torch.sum(target * logprobs, dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class PretrainSentLoss(torch.nn.Module):
    def __init__(self, base_criterion: torch.nn.Module, loss_type: str, args=None, 
        distill_type='none', alpha=0., beta=0., tau=0., set_training_mode=False):
        super().__init__()
        self.base_criterion = base_criterion
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        assert distill_type in ['none', 'feat', 'logits', 'logits_kl']
        self.distill_type:str = distill_type
        if beta > 0:
            assert self.distill_type.startswith("logits")
            teacher_model = args.teacher_model if args.teacher_model else args.model
            self.teacher_model = create_model(
                    teacher_model,
                    pretrained=args.pretrained,
                    num_classes=args.nb_classes,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path,
                    drop_block_rate=None,
                    dataset=None,
                    args=args
                )
            if args.teacher_path:
                self.teacher_model.initialize_parameters(args.teacher_path)
            device = torch.device(args.device)
            self.teacher_model.to(device)
            self.teacher_model.requires_grad_(False)
            self.fp32 = args.fp32_resume
            self.set_training_mode=set_training_mode

    def forward(self, inputs, outputs, labels: torch.Tensor):
        if isinstance(outputs, torch.Tensor):
            loss = self.base_criterion(outputs, labels)
            return loss
        # assume that the model outputs a tuple of outputs
        if self.alpha > 0.:
            assert self.distill_type.startswith("feat")
            # assume that the model outputs a tuple of [outputs1, outputs2, distill_loss]
            outputs1, outputs2, distill_loss = outputs
            distill_loss = torch.mean(distill_loss)
        else:
            # assume that the model outputs a tuple of [outputs1, outputs2]
            outputs1, outputs2 = outputs
            distill_loss = 0.
        if self.loss_type in ["softCE", "smoothCE"]:
            labels = labels / torch.sum(labels, dim=1, keepdim=True)
        loss1 = self.base_criterion(outputs1, labels)
        loss2 = self.base_criterion(outputs2, labels)
        base_loss = (loss1 + loss2) / 2.0
        loss = (1 - self.alpha) * base_loss + self.alpha * distill_loss
        if self.beta > 0:
            self.teacher_model.train(self.set_training_mode)
            teacher_outputs1, teacher_outputs2 = self.teacher_model(inputs)
            teacher_outputs1, teacher_outputs2 = teacher_outputs1.detach(), teacher_outputs2.detach()
            if self.distill_type == 'logits_kl':
                distill_loss1 = kl_div(outputs1, teacher_outputs1, T=self.tau)
                distill_loss2 = kl_div(outputs2, teacher_outputs2, T=self.tau)
                distill_loss = (distill_loss1 + distill_loss2) / 2.0
            else:
                assert self.distill_type == "logits"
                distill_loss1 = cross_entropy(outputs1, teacher_outputs1)
                distill_loss2 = cross_entropy(outputs2, teacher_outputs2)
                distill_loss = (distill_loss1 + distill_loss2) / 2.0
            loss = (1 - self.beta) * loss + self.beta * distill_loss
        return loss, distill_loss


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
