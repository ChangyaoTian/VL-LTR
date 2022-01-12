# 8 GPU
cfg = dict(
    model='LGR_vit16',
    desc_path='data/places',
    pretrained_clip='pretrained/ViT-B-16.pt',
    context_length=75,
    pretrain_cvlp=False,
    pretrain_cvlp_path='checkpoints/places_pretrain_vit16/',
    loss_type="CE",
    two_branch=True,

    use_mcloader=True,
    data_set='PLACES_LT',
    drop_last=True,

    weight_sample=True,
    use_sqrt_freq=True,

    lr=5e-5,
    min_lr=0,
    warmup_epochs=0,
    text_lr=1e-6,

    epochs=50,
    batch_size=128,

    repeated_aug=False,
    clip_ms=True,
    test=True,
)
