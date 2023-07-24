import os
import random
import shutil
from shutil import copy2
from tqdm import tqdm


def spilt_datasets(data_root, trainDir, validDir):
    trainfiles = os.listdir(data_root)
    num_train = len(trainfiles)
    print("num_train: " + str(num_train))
    index_list = list(range(num_train))
    # print(index_list)
    random.shuffle(index_list)
    num = 0
    # trainDir = '../data/2023/extra_train/images'
    # validDir = '../data/2023/extra_val/images'
    for i in index_list:
        fileName = os.path.join(data_root, trainfiles[i])
        if num < num_train * 0.8:
            print(str(fileName))
            copy2(fileName, trainDir)
        else:
            copy2(fileName, validDir)
        num += 1


def cp_labels(lable_root, trainDir, validDir):
    # lable_root = '../data/2023/extra/labels'

    # trainDir = '../data/2023/extra_train/images'
    trainfiles = os.listdir(trainDir)

    # validDir = '../data/2023/extra_val/images'
    valfiles = os.listdir(validDir)

    for i in tqdm(trainfiles):
        fileName = os.path.join(lable_root, i)
        fileName = fileName.replace('.tif', '.txt')
        copy2(fileName, '/workspace/pycharm_project/mmrotate/data/2023/train/labels')

    for j in tqdm(valfiles):
        fileName = os.path.join(lable_root, j)
        fileName = fileName.replace('.tif', '.txt')
        copy2(fileName, '/workspace/pycharm_project/mmrotate/data/2023/val/labels')


def main():
    data_root = '/workspace/pycharm_project/mmrotate/data/2023/full_train/images/'
    trainDir = '/workspace/pycharm_project/mmrotate/data/2023/train/images/'
    validDir = '/workspace/pycharm_project/mmrotate/data/2023/val/images'
    # spilt_datasets(data_root, trainDir, validDir)

    lable_root = '/workspace/pycharm_project/mmrotate/data/2023/full_train/new_labels/'
    cp_labels(lable_root, trainDir, validDir)


if __name__ == '__main__':
    main()
