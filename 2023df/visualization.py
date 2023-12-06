# Copyright (c) 2023 ✨Challyfilio✨
# 翻转图像、可视化
import os
import csv
import cv2
import numpy as np
import random

from tqdm import tqdm

from convert_res import get_file_basename

# data_root = '/workspace/pycharm_project/mmrotate/wxytests/data/'
# image_root = '/workspace/pycharm_project/Dataset/train_overlap/images/'
# label_root = '#'

data_root = '/workspace/pycharm_project/Dataset/extra_original/'
image_root = data_root + 'images/'
label_root = data_root + 'annfiles/'


# 翻转图像
def filp(img_path, label_path, filp_mode):
    if filp_mode == -1 or filp_mode == 0 or filp_mode == 1:
        img = cv2.imread(img_path)
        img = cv2.flip(img, filp_mode)  # -1,0,1
        bsname = get_file_basename(img_path)
        cv2.imwrite(image_root + bsname + '_' + str(filp_mode) + '.tif', img)  # 存图
        if label_path == '#':
            pass
        else:
            with open(label_path, "r", encoding='utf-8') as f:
                lines = f.readlines()
                for l in lines:
                    data = []
                    for i in range(0, 8):
                        position = l.split(' ')[i]
                        data.append(float(position))
                    data.append(l.split(' ')[-1])
                    if filp_mode == 0:  # v
                        data[1] = 1024.0 - data[1]
                        data[3] = 1024.0 - data[3]
                        data[5] = 1024.0 - data[5]
                        data[7] = 1024.0 - data[7]
                    elif filp_mode == 1:  # h
                        data[0] = 1024.0 - data[0]
                        data[2] = 1024.0 - data[2]
                        data[4] = 1024.0 - data[4]
                        data[6] = 1024.0 - data[6]
                    elif filp_mode == -1:  # hv
                        data[1] = 1024.0 - data[1]
                        data[3] = 1024.0 - data[3]
                        data[5] = 1024.0 - data[5]
                        data[7] = 1024.0 - data[7]
                        data[0] = 1024.0 - data[0]
                        data[2] = 1024.0 - data[2]
                        data[4] = 1024.0 - data[4]
                        data[6] = 1024.0 - data[6]
                    else:
                        pass
                    with open(label_root + bsname + '_' + str(filp_mode) + '.txt', "a", encoding='utf-8') as f_out:
                        out_str = ''
                        for i in range(0, 8):
                            out_str = out_str + str(data[i]) + ' '
                        out_str = out_str + data[8]
                        f_out.write(out_str)
    else:
        pass


def process_dataset():
    files = os.listdir(image_root)
    for f in tqdm(files):
        imagePath = os.path.join(image_root, f)
        labelPath = os.path.join(label_root, f)
        labelPath = labelPath.replace('.tif', '.txt')
        mode_list = [-1, 0, 1]
        mode = random.choice(mode_list)
        # print(imagePath)
        # print(labelPath)
        # print(mode)
        filp(imagePath, labelPath, mode)
        # filp(imagePath, '#', mode)
    print('finish')


def vvv(img_path, txt_path):
    count = 0
    with open(txt_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for l in lines:
            if os.path.exists('out.jpg'):
                img = cv2.imread('out.jpg')
            else:
                img = cv2.imread(img_path)
            data = []
            for i in range(0, 8):
                position = l.split(' ')[i]
                data.append(float(position))

            cnt = np.array([
                [[data[0], data[1]]],
                [[data[2], data[3]]],
                [[data[4], data[5]]],
                [[data[6], data[7]]]
            ], dtype=np.float32)
            # print("shape of cnt: {}".format(cnt.shape))
            rect = cv2.minAreaRect(cnt)
            # print("rect: {}".format(rect))

            # the order of the box points: bottom left, top left, top right,
            # bottom right
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            cv2.imwrite('out.jpg', img)
            count += 1
    print(count)
    exit()


def vvvvv(mode):
    img_path = '/workspace/pycharm_project/mmrotate/data/2023/train/images/run2_train_00003.tif'
    txt_path = '/workspace/pycharm_project/mmrotate/data/2023/train/labels/run2_train_00003.txt'
    count = 0
    with open(txt_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for l in lines:
            if os.path.exists('out.jpg'):
                img = cv2.imread('out.jpg')
            else:
                img = cv2.imread(img_path)
                img = cv2.flip(img, mode)  # y
            data = []
            for i in range(0, 8):
                position = l.split(' ')[i]
                data.append(float(position))
            if mode == 0:  # v
                data[1] = 1024.0 - data[1]
                data[3] = 1024.0 - data[3]
                data[5] = 1024.0 - data[5]
                data[7] = 1024.0 - data[7]
            elif mode == 1:  # h
                data[0] = 1024.0 - data[0]
                data[2] = 1024.0 - data[2]
                data[4] = 1024.0 - data[4]
                data[6] = 1024.0 - data[6]
            elif mode == -1:  # hv
                data[1] = 1024.0 - data[1]
                data[3] = 1024.0 - data[3]
                data[5] = 1024.0 - data[5]
                data[7] = 1024.0 - data[7]
                data[0] = 1024.0 - data[0]
                data[2] = 1024.0 - data[2]
                data[4] = 1024.0 - data[4]
                data[6] = 1024.0 - data[6]
            else:
                pass

            cnt = np.array([
                [[data[0], data[1]]],
                [[data[2], data[3]]],
                [[data[4], data[5]]],
                [[data[6], data[7]]]
            ], dtype=np.float32)
            # print("shape of cnt: {}".format(cnt.shape))
            rect = cv2.minAreaRect(cnt)
            # print("rect: {}".format(rect))

            # the order of the box points: bottom left, top left, top right,
            # bottom right
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            cv2.imwrite('out.jpg', img)
            count += 1
    print(count)
    exit()

    img = cv2.imread(img_path)
    row = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    cnt = np.array([
        [[row[0], row[1]]],
        [[row[2], row[3]]],
        [[row[4], row[5]]],
        [[row[6], row[7]]]
    ], dtype=np.float32)
    # print("shape of cnt: {}".format(cnt.shape))
    rect = cv2.minAreaRect(cnt)
    # print("rect: {}".format(rect))

    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    cv2.imwrite('out.jpg', img)


def draw_box():
    tests_root = '/workspace/pycharm_project/mmrotate/data/2023/test/images/'
    outdir = './visuallzation/'
    with open('/workspace/pycharm_project/mmrotate/2023df/post_classification/example_swin_ft_0.46.csv',
              encoding='utf-8-sig') as f:
        count = 0
        for row in csv.reader(f, skipinitialspace=True):
            print(row)
            file_name = row[0]
            label = row[1]
            # bbox = []
            # for i in range(3, 11):
            #     bbox.append(int(row[i]))
            if os.path.exists(outdir + file_name.replace('.tif', '') + '.jpg'):
                img = cv2.imread(outdir + file_name.replace('.tif', '') + '.jpg')
            else:
                img = cv2.imread(tests_root + file_name)

            cnt = np.array([
                [[row[3], row[4]]],
                [[row[5], row[6]]],
                [[row[7], row[8]]],
                [[row[9], row[10]]]
            ], dtype=np.float32)
            # print("shape of cnt: {}".format(cnt.shape))
            rect = cv2.minAreaRect(cnt)
            # print("rect: {}".format(rect))

            # the order of the box points: bottom left, top left, top right,
            # bottom right
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            cv2.putText(img, label, (int(eval(row[3])), int(eval(row[4]))), font, 1, (0, 0, 255), 1)

            cv2.imwrite(outdir + file_name.replace('.tif', '') + '.jpg', img)
            count += 1
            # if count == 100:
            #     break


if __name__ == "__main__":
    # img_path = '/workspace/pycharm_project/mmrotate/data/2023/train/images/run2_train_00039_-1.tif'
    # txt_path = '/workspace/pycharm_project/mmrotate/data/2023/train/labels/run2_train_00039_-1.txt'
    # # filp(img_path, txt_path, -1)
    # vvv(img_path, txt_path)

    process_dataset()
