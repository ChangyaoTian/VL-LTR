# 8 GPU
cfg = dict(
    model='CVLP_r50',
    desc_path='data/iNat',
    pretrained_clip='pretrained/RN50.pt',
    context_length=75,
    pretrain_cvlp=True,
    loss_type="smoothCE",

    data_set='INAT',
    drop_last=True,
    eval_pretrain=True,

    weight_sample=True,
    use_sqrt_freq=True,

    lr=5e-4,
    epochs=100,
    batch_size=256,

    repeated_aug=False,
    mixup=0.,
    cutmix=0.,
    clip_ms=True,
    distillation_beta=0.5,
    distillation_type='logits',
)
