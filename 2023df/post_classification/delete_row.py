import os
import csv
from tqdm import tqdm


def main():
    test_data_root = '/workspace/pycharm_project/mmrotate/data/2023/test/images/'
    csv_root = '/workspace/pycharm_project/mmrotate/2023df/'
    csv_file = os.path.join(csv_root, 'example_extra_ft_1.csv')

    with open('out4_del.csv', 'w') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(['ImageID', 'LabelName', 'Conf', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4'])

    with open(csv_file, encoding='utf-8-sig') as f:
        for row in tqdm(csv.reader(f, skipinitialspace=True)):
            print(row)
            if float(row[2]) < 0.3:
                pass
            else:
                with open('out4_del.csv', 'a', newline='') as csv_f:
                    # with open('example.csv', 'w') as csv_file:
                    writer = csv.writer(csv_f)
                    writer.writerow(row)
    print("finish")


if __name__ == '__main__':
    main()
