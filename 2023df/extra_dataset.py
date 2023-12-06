# Copyright (c) 2023 ✨Challyfilio✨
# 处理比赛补充数据集标签 csv --> txt
import os
import csv

if __name__ == '__main__':
    label_root = '/workspace/pycharm_project/Dataset/extra/annfiles/'
    csv_path = '/workspace/pycharm_project/Dataset/extra/train_labels_final.csv'
    count = 0
    with open(csv_path, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):  # 读行
            # print(row)
            filename = row[0].split('.')[0]
            # print(filename)
            data = ''
            for i in range(2, 10):
                data += str(row[i]) + ' '
            data += row[1] + '\n'
            # print(data)
            # print(filename)
            with open(label_root + filename + '.txt', "a", encoding='utf-8') as f1:
                f1.write(data)
            count += 1
            # if os.path.exists(label_root + filename + '.txt'):
            #     with open(label_root + filename + '.txt', "w", encoding='utf-8') as f:
            #         f.write(data)
            # else:
            #     with open(label_root + filename + '.txt', "w", encoding='utf-8') as f:
            #         f.write(data)
    print(count)
