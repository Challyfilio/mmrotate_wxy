测试
```shell
python ./tools/test.py configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90.py train_swin_ft_1.2_fl_nms0.7/epoch_12.pth --format-only --eval-options submission_dir=test_swin_ft_1.2_f1_nms0.7_test0.7/Task1_results
```
```shell
python ./tools/test.py configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90.py work_dirs_swin_ft/epoch_12.pth --format-only --eval-options submission_dir=work_dirs_swin_ft_0.3/Task1_results
```
```shell
python ./tools/test.py configs/oriented_rcnn/oriented_rcnn_swin_tiny_fpn_1x_dota_le90.py work_dir/oriented_rcnn_swin_ms_ft_lsce_cslfpn2_nms0.8/epoch_11.pth --format-only --eval-options submission_dir=test_dir/test_oriented_rcnn_swin_ms_ft_lsce_cslfpn2_nms0.8_test0.9/Task1_results
```

训练

```shell
python tools/train.py configs/oriented_rcnn/oriented_rcnn_swin_tiny_fpn_1x_dota_le90.py --work-dir oriented_rcnn_swin_extra
```
```shell
python tools/train.py configs/oriented_rcnn/oriented_rcnn_swin_tiny_fpn_1x_dota_le90.py --load-from work_dir/oriented_rcnn_swin_extra/epoch_12.pth --work-dir work_dir/oriented_rcnn_swin_ms_ft_lsce_cslfpn2_nms0.8
```
```shell
python tools/train.py configs/oriented_rcnn/oriented_rcnn_swin_tiny_fpn_1x_dota_le90.py --resume-from work_dir/oriented_rcnn_swin_ms_ft_lsce_cslfpn2_nms0.8/epoch_9.pth --work-dir work_dir/oriented_rcnn_swin_ms_ft_lsce_cslfpn2_nms0.8
```
swin
```shell
python tools/train.py configs/redet/redet_re50_refpn_1x_dota_ms_rr_le90.py --no-validate --work-dir work_dirs_redet_extra_ms
```
```shell
python tools/train.py configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90.py --resume-from train_swin_ft_1.2_fl_nms0.7/epoch_5.pth --no-validate --work-dir train_swin_ft_1.2_fl_nms0.7
```

数据集划分

```shell
python tools/data/dota2023/split/img_split.py --base-json tools/data/dota2023/split/split_configs/ss_train.json
```
```shell
python tools/data/dota2023/split/img_split.py --base-json tools/data/dota2023/split/split_configs/ms_trainval.json
```