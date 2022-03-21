# 8 GPU
cfg = dict(
    model='LGR_r50_v_detach_img_grad',
    desc_path='data/iNat',
    pretrained_clip='pretrained/RN50.pt',
    context_length=75,
    pretrain_cvlp=False,
    pretrain_cvlp_path='checkpoints/inat_pretrain_r50/',
    loss_type="CE",
    two_branch=True,

    data_set='INAT',
    drop_last=True,

    weight_sample=True,
    use_sqrt_freq=True,

    lr=2e-5,
    min_lr=0,
    warmup_epochs=0,
    text_lr=1e-6,

    epochs=360,
    batch_size=128,

    repeated_aug=False,
    clip_ms=True,
)
