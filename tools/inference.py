# 导入os模块
import os

'''
python demo/image_demo.py \
    demo/demo.jpg oriented_rcnn_r50_fpn_1x_dota_le90.py \
    oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth \
    --out-file result.jpg
'''

# 定义一个函数，接受四个参数
def vis_rotated_result(image_path, config_path, checkpoint_path, outfile_path):
    # 拼接命令字符串
    cmd = 'python ../demo/image_demo.py' + ' ' + image_path + ' ' + config_path + ' ' + checkpoint_path + ' --out-file ' + outfile_path
    # 执行命令字符串
    os.system(cmd)

if __name__ == '__main__':
    image_path = '/workspace/pycharm_project/mmrotate/demo/dota_demo.jpg'
    config_path = '/workspace/pycharm_project/mmrotate/configs/oriented_rcnn/oriented_rcnn_swin_tiny_fpn_1x_dota_le90.py'
    checkpoint_path = '/workspace/pycharm_project/mmrotate/work_dir/oriented_rcnn_swin_tiny_fpn_ce_dota_ss_baseline/epoch_12.pth'
    outfile_path = '/workspace/pycharm_project/mmrotate/demo/result1220.jpg'

    vis_rotated_result(image_path, config_path, checkpoint_path, outfile_path)
