import csv
from tqdm import tqdm

def SymbolJudgment(input_list:list):
    Symbol_str = ''
    count_0 = 0
    count_1024 = 0
    for pos in input_list:
        if float(pos) >= 0 and float(pos) <= 1024:
            Symbol_str += '1'
        elif float(pos) < 0:
            Symbol_str += '0'
            count_0 += 1
        elif float(pos) > 1024:
            Symbol_str += '2'
            count_1024 += 1
        else:
            pass
    return Symbol_str,count_0,count_1024


#  0, 1, 2, 3, 4, 5, 6, 7
# x1,y1,x2,y2,x3,y3,x4,y4
def CalculateCoordinates(Symbol_str:str,pos_list:list):
    # list 转 float
    plf = []
    for i in pos_list:
        plf.append(float(i))
    if Symbol_str == '10101111':
        new_x1 = Calculate_XY(plf[6],plf[0],plf[7],plf[1])
        new_x2 = Calculate_XY(plf[4],plf[2],plf[5],plf[3])
        plf[0]+=new_x1
        plf[1]=0
        plf[2]+=new_x2
        plf[3]=0
    elif Symbol_str == '11101011':
        new_x2 = Calculate_XY(plf[0],plf[2],plf[1],plf[3])
        new_x3 = Calculate_XY(plf[6],plf[4],plf[7],plf[5])
        plf[2]+=new_x2
        plf[3]=0
        plf[4]+=new_x3
        plf[5]=0
    elif Symbol_str == '01011111':
        new_y1 = Calculate_XY(plf[7],plf[1],plf[6],plf[0])
        new_y2 = Calculate_XY(plf[5],plf[3],plf[4],plf[2])
        plf[0]=0
        plf[1]+=new_y1
        plf[2]=0
        plf[3]+=new_y2
    elif Symbol_str == '01111101':
        new_y1 = Calculate_XY(plf[3],plf[1],plf[2],plf[0])
        new_y4 = Calculate_XY(plf[5],plf[7],plf[4],plf[6])
        plf[0]=0
        plf[1]+=new_y1
        plf[6]=0
        plf[7]+=new_y4
    # elif Symbol_str == '11111212':
    #     new_x3 = Calculate_XY2(plf[2],plf[4],plf[5],plf[3])
    #     new_x4 = Calculate_XY2(plf[0],plf[6],plf[7],plf[1])
    #     plf[4]+=new_x3
    #     plf[5]=1024
    #     plf[6]+=new_x4
    #     plf[7]=1024
    # elif Symbol_str == '12111112':
    #     new_x1 = Calculate_XY2(plf[2],plf[0],plf[1],plf[3])
    #     new_x4 = Calculate_XY2(plf[4],plf[6],plf[7],plf[5])
    #     plf[0]+=new_x1
    #     plf[1]=1024
    #     plf[6]+=new_x4
    #     plf[7]=1024
    # elif Symbol_str == '11112121':
    #     new_y3 = Calculate_XY3(plf[3],plf[5],plf[4],plf[2])
    #     new_y4 = Calculate_XY3(plf[1],plf[7],plf[6],plf[0])
    #     plf[4]=1024
    #     plf[5]+=new_y3
    #     plf[6]=1024
    #     plf[7]+=new_y4
    # elif Symbol_str == '11212111':
    #     new_y2 = Calculate_XY3(plf[1],plf[3],plf[2],plf[0])
    #     new_y3 = Calculate_XY3(plf[7],plf[5],plf[4],plf[6])
    #     plf[2]=1024
    #     plf[3]+=new_y2
    #     plf[4]=1024
    #     plf[5]+=new_y3
    else:
        pass
    return plf


def Calculate_XY(param1,param2,param3,param4):
    return (param1-param2)*abs(param4)/(param3-param4)


def Calculate_XY2(param1,param2,param3,param4):
    return (param1-param2)*(1024-param4)/(param3-param4)


def Calculate_XY3(param1,param2,param3,param4):
    return (param1-param2)*(param3-1024)/(param3-param4)


def Process_Pos(pos_list:list):
    for i in range(0,len(pos_list)):
        if pos_list[i]<0:
            pos_list[i]=0
        elif pos_list[i]>1024:
            pos_list[i]=1024
        else:
            pass
    return pos_list


def main(input_csv_path,output_csv_path):
    with open(output_csv_path, 'w') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(['ImageID', 'LabelName', 'Conf', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4'])

    with open(input_csv_path, encoding='utf-8-sig') as f:
        for row in tqdm(csv.reader(f, skipinitialspace=True)):
            final_data = []
            # print(row)
            tif_name = row[0]
            clss = row[1]
            conf = row[2]
            pos_list = []
            for i in range(3, 11):
                pos_list.append(float(row[i]))
            Symbol_str,count_0,count_1024 = SymbolJudgment(pos_list)
            if count_0 == 2 or count_1024 == 2:
                # print(pos_list)
                pos_list = CalculateCoordinates(Symbol_str,pos_list)
                # pass
            elif count_0 == 1 or count_1024 == 1:
                # 一个点在外面
                pos_list = Process_Pos(pos_list)
                # pass
            else:
                pass
            final_data.append(tif_name)
            final_data.append(clss)
            final_data.append(conf)
            final_data = final_data + pos_list
            # print(final_data)
            with open(output_csv_path, 'a', newline='') as csv_f:
                # with open('example.csv', 'w') as csv_file:
                writer = csv.writer(csv_f)
                writer.writerow(final_data)
    print("finish")


if __name__ == '__main__':
    input_csv_path = './oriented_rcnn_swin_ms_ft_lsce_cslfpn2_nms0.8_test0.9.csv'
    output_csv_path = './oriented_rcnn_swin_ms_ft_lsce_cslfpn2_nms0.8_test0.9_cropped.csv'
    main(input_csv_path,output_csv_path)


    # # input_list = ['-7.56','298.48','103.61','273.67','109.32','299.27','-1.85','324.08']
    # # input_list = ['492.26','-2.16','512.28','-5.94','516.36','15.64','496.34','19.42']
    # input_list = ['998.40','371.72','1024.41','366.63','1028.99','390.04','1002.98','395.13']
    # Symbol_str,count_0,count_1024 = SymbolJudgment(input_list)
    # print(SymbolJudgment(input_list))
    # if count_0 == 2 or count_1024 == 2:
    #     print(input_list)
    #     print(CalculateCoordinates(Symbol_str,input_list))
    # elif count_0 == 1 or count_1024 == 1:
    #     # 一个点在外面
    #     pass
    # else:
    #     pass