# _base_ = ['./oriented_rcnn_r50_fpn_1x_dota_le90.py']
_base_ = ['./oriented_rcnn_1x_le90.py']

fp16 = dict(loss_scale='dynamic')