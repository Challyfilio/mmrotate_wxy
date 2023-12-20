_base_ = ['./oriented_rcnn_r50_fpn_1x_dota_le90.py']

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    # pretrained='/workspace/pycharm_project/mmrotate/pretrain/rsp-swin-t-ckpt.pth',
    backbone=dict(
        _delete_=True,
        type='ViT_Win_RVSA_V3_WSZ7',
        img_size=1024,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.15,
        use_abs_pos_emb=True,
        ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        num_outs=5),
        )

# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))

# lr_config = dict(step=[8, 11])

optimizer = dict(
    _delete_=True, 
    type='AdamW', 
    lr=0.0001, 
    betas=(0.9, 0.999), 
    weight_decay=0.05,
    paramwise_cfg=dict(
        num_layers=12, 
        layer_decay_rate=0.75,
        custom_keys={
            'bias': dict(decay_multi=0.),
            'pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            "rel_pos_h": dict(decay_mult=0.),
            "rel_pos_w": dict(decay_mult=0.),
        }))

lr_config = dict(step=[8, 11])