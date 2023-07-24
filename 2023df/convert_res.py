import os
import csv


def find_files_with_suffix(folder_path, suffix):
    res_file_list = []
    all_files = os.listdir(folder_path)
    filtered_files = [file for file in all_files if file.endswith(suffix)]
    for j in filtered_files:
        res_file_list.append(os.path.join(folder_path, j))
    return res_file_list


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def main(data_root, out_file):
    list = find_files_with_suffix(data_root, '.txt')
    with open(out_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['ImageID', 'LabelName', 'Conf', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4'])
    for file in list:
        class_name = get_file_basename(file).split('_')[1]
        with open(file, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for l in lines:
                data = []
                data.append(l.split(' ')[0] + '.tif')
                data.append(class_name)
                for i in range(1, 9):
                    position = l.split(' ')[i]
                    # if '-' in position:
                    #     position = '0'
                    data.append(position)
                data.append(l.split(' ')[9].replace('\n', ''))
                with open(out_file, 'a', newline='') as csv_file:
                    # with open('example.csv', 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(data)
                # print(data)
    # exit()
    #
    # print(len(list))


if __name__ == '__main__':
    data_root = '../test_dir/test_oriented_rcnn_swin_ms_ft_lsce_cslfpn2_nms0.8_test0.9/Task1_results'
    out_file = 'oriented_rcnn_swin_ms_ft_lsce_cslfpn2_nms0.8_test0.9.csv'
    main(data_root, out_file)
