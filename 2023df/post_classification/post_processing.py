import os
import csv
import json
from tqdm import tqdm


def label_trans(cls_label: str):
    # convert cls --> det
    DET_CLASSES = ['F5', 'P7', 'F2', 'W1', 'S4', 'T1', 'C14', 'B3', 'A7', 'A8', 'C2', 'P3', 'F8', 'C8', 'W2', 'S7',
                   'C13', 'T7', 'L3', 'Y1', 'M2', 'S5', 'V1', 'T2', 'S6', 'C10', 'S1', 'R2', 'D2', 'V2', 'C9', 'P2',
                   'H1', 'U2', 'H3', 'N1', 'T5', 'A9', 'D1', 'C6', 'C5', 'T8', 'P5', 'K2', 'P4', 'H2', 'A3', 'B1', 'E2',
                   'K3', 'C12', 'C15', 'L4', 'S2', 'R1', 'W3', 'T9', 'C11', 'M5', 'E4', 'R3', 'F7', 'U1', 'C3', 'K1',
                   'M1', 'A6', 'F3', 'E3', 'C1', 'B2', 'T6', 'P1', 'K5', 'K4', 'A4', 'L2', 'C16', 'S3', 'C4', 'A5',
                   'I1', 'A1', 'E1', 'P6', 'F6', 'C7', 'M4', 'F1', 'T10', 'T3', 'L1', 'Z1', 'A2', 'T4', 'M3', 'R4',
                   'T11']

    CLS_CLASSES = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B1', 'B2', 'B3', 'C1', 'C10', 'C11', 'C12',
                   'C13', 'C14', 'C15', 'C16', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D1', 'D2', 'E1', 'E2',
                   'E3', 'E4', 'F1', 'F2', 'F3', 'F5', 'F6', 'F7', 'F8', 'H1', 'H2', 'H3', 'I1', 'K1', 'K2', 'K3', 'K4',
                   'K5', 'L1', 'L2', 'L3', 'L4', 'M1', 'M2', 'M3', 'M4', 'M5', 'N1', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6',
                   'P7', 'R1', 'R2', 'R3', 'R4', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'T1', 'T10', 'T11', 'T2',
                   'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'U1', 'U2', 'V1', 'V2', 'W1', 'W2', 'W3', 'Y1', 'Z1']
    # print(len(DET_CLASSES))
    # print(len(CLS_CLASSES))

    index = CLS_CLASSES.index(cls_label)
    return DET_CLASSES[index]


def read_json():
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    class_list = []
    for i in range(0, 98):
        class_list.append(class_indict[str(i)])
    return class_list


def main():
    csv_root = '/workspace/pycharm_project/mmrotate/2023df/'
    csv_file = os.path.join(csv_root, 'example_extra_ft_1.csv')
    with open('out2.csv', 'w') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(['ImageID', 'LabelName', 'Conf', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4'])
    with open('out1.csv', encoding='utf-8-sig') as f:
        for row in tqdm(csv.reader(f, skipinitialspace=True)):
            final_data = []
            # print(row)
            tif_name = row[0]
            cls_label = row[1]
            det_cls = label_trans(cls_label)
            conf = row[2]
            # tif_path = os.path.join(test_data_root, tif_name)  # 图像路径
            pos_list = []
            for i in range(3, 11):
                pos_list.append(float(row[i]))
            # outimg, cls = crop(tif_path, pos_list, cls)  # 剪裁图像
            # new_cls_index = predict_single(outimg)  # 图像分类预测
            # new_cls = label_trans(new_cls_index)
            final_data.append(tif_name)
            final_data.append(det_cls)
            final_data.append(conf)
            final_data = final_data + pos_list
            with open('out2.csv', 'a', newline='') as csv_f:
                # with open('example.csv', 'w') as csv_file:
                writer = csv.writer(csv_f)
                writer.writerow(final_data)
            # exit()
    print("finish")


if __name__ == '__main__':
    main()
