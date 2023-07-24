import os
import csv
from tqdm import tqdm

from predict import predict_single
from corp_img import crop


def label_trans(index: int):
    CLASSES = (
        'F5', 'P7', 'F2', 'W1', 'S4', 'T1', 'C14', 'B3', 'A7', 'A8', 'C2', 'P3', 'F8', 'C8', 'W2', 'S7', 'C13', 'T7',
        'L3',
        'Y1', 'M2', 'S5', 'V1', 'T2', 'S6', 'C10', 'S1', 'R2', 'D2', 'V2', 'C9', 'P2', 'H1', 'U2', 'H3', 'N1', 'T5',
        'A9',
        'D1', 'C6', 'C5', 'T8', 'P5', 'K2', 'P4', 'H2', 'A3', 'B1', 'E2', 'K3', 'C12', 'C15', 'L4', 'S2', 'R1', 'W3',
        'T9',
        'C11', 'M5', 'E4', 'R3', 'F7', 'U1', 'C3', 'K1', 'M1', 'A6', 'F3', 'E3', 'C1', 'B2', 'T6', 'P1', 'K5', 'K4',
        'A4',
        'L2', 'C16', 'S3', 'C4', 'A5', 'I1', 'A1', 'E1', 'P6', 'F6', 'C7', 'M4', 'F1', 'T10', 'T3', 'L1', 'Z1', 'A2',
        'T4',
        'M3', 'R4', 'T11')
    return CLASSES[index]


def main():
    test_data_root = '/workspace/pycharm_project/mmrotate/data/2023/test/images/'
    csv_root = '/workspace/pycharm_project/mmrotate/2023df/'
    csv_file = os.path.join(csv_root, 'example_swin_ft_0.46.csv')

    with open('out3.csv', 'w') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(['ImageID', 'LabelName', 'Conf', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4'])

    with open(csv_file, encoding='utf-8-sig') as f:
        for row in tqdm(csv.reader(f, skipinitialspace=True)):
            final_data = []
            # print(row)
            tif_name = row[0]
            cls = row[1]
            # conf = row[2]
            tif_path = os.path.join(test_data_root, tif_name)  # 图像路径
            pos_list = []
            for i in range(3, 11):
                pos_list.append(float(row[i]))
            outimg, _ = crop(tif_path, pos_list, cls)  # 剪裁图像
            new_cls, new_conf = predict_single(outimg)  # 图像分类预测,输出：类别，置信度
            # new_cls = label_trans(new_cls_index)
            final_data.append(tif_name)
            final_data.append(new_cls)
            final_data.append(new_conf)
            final_data = final_data + pos_list
            # print(final_data)
            with open('out3.csv', 'a', newline='') as csv_f:
                # with open('example.csv', 'w') as csv_file:
                writer = csv.writer(csv_f)
                writer.writerow(final_data)
    print("finish")


if __name__ == '__main__':
    main()
