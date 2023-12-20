# Copyright (c) 2023 ✨Challyfilio✨
import torch
from loguru import logger

pretrained_A = '/workspace/pycharm_project/mmpretrain/pretrain/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192_20220829-0e15782d.pth'  # A
pretrained_B = '/workspace/pycharm_project/mmrotate/work_dir/oriented_rcnn_swin_b_fpn_ce_dota_ss_nms0.8/epoch_12.pth'  # bs B

pretrained_A = '/workspace/pycharm_project/mmrotate/pretrain/rsp-swin-t-ckpt.pth'
pretrained_B = '/workspace/pycharm_project/mmrotate/pretrain/swin_tiny_patch4_window7_224.pth'
# A-->B

if __name__ == '__main__':
    pretrained_dict_A = torch.load(pretrained_A)
    print(pretrained_dict_A.keys())
    print(pretrained_dict_A['model'].keys())
    # print(pretrained_dict_A['state_dict']['backbone.patch_embed.projection.bias'])
    pretrained_dict_B = torch.load(pretrained_B)
    print(pretrained_dict_B.keys())
    print(pretrained_dict_B['model'].keys())
    # print(pretrained_dict_B['state_dict']['backbone.patch_embed.projection.bias'])
    exit()

    count = 0
    for a_key in list(pretrained_dict_A['state_dict'].keys()):
        for b_key in list(pretrained_dict_B['state_dict'].keys()):
            if a_key == b_key:
                if pretrained_dict_B['state_dict'][b_key].shape == pretrained_dict_A['state_dict'][a_key].shape:
                    # logger.info(a_key)
                    pretrained_dict_B['state_dict'][b_key] = pretrained_dict_A['state_dict'][a_key]
                    count += 1
                else:
                    pass
    print(count)
    # print(pretrained_dict_B.keys())
    # print(len(pretrained_dict_B['state_dict'].keys()))
    torch.save(pretrained_dict_B, "my_model.pth")
    logger.success('finish')
