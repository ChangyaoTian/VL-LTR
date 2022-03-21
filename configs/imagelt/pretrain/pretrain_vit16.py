# 8 GPU
cfg = dict(
    model='CVLP_vit16',
    desc_path='data/imagenet',
    pretrained_clip='pretrained/ViT-B-16.pt',
    context_length=75,
    pretrain_cvlp=True,
    loss_type="smoothCE",

    data_set='IMNET_LT',
    drop_last=True,

    weight_sample=True,
    use_sqrt_freq=True,
    train_mode=False,

    lr=3.5e-5,
    min_lr=0.,

    epochs=50,
    batch_size=128,

    repeated_aug=False,
    mixup=0.,
    cutmix=0.,
    clip_ms=True,
    distillation_beta=0.5,
    distillation_type='logits',
    
    eval_pretrain=True,
    test=True
)
