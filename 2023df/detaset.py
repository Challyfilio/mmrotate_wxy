import os

file_path = "/workspace/pycharm_project/mmrotate/data/2023/extra/labels"
# 获取file_path路径下的所有TXT文本内容和文件名
def get_text_list(file_path):
    files = os.listdir(file_path)
    text_list = []
    for file in files:
        with open(os.path.join(file_path, file), "r", encoding="UTF-8") as f:
            text_list.append(f.read())
    return text_list, files
#print(get_text_list(file_path))
files= os.listdir(file_path) #得到文件夹下的所有文件名称
txts = []
for file in files: #遍历文件夹
    position = file_path+'/'+ file #构造绝对路径，"\\"，其中一个'\'为转义符
    #print (position)
    with open(position, "r",encoding='utf-8') as f:    #打开文件
        data = f.read()   #读取文件
        txts.append(data)
txts = ' '.join(txts)#转化为非数组类型
#print(data)
#print (txts)
lable = str.split(txts)

#print(lable)
i = 8
L = []
while i < len(lable):
    L.append(lable[i])
    #print(f'{lable[i]}')
    i += 9
#print(L)
res = []
for i in L:
    if i not in res:
        res.append(i)
print(res)
print(len(res))
filename = ('output.txt')
outfile = open(filename, 'w')
outfile.writelines([str(i)+'\n' for i in res])
outfile.close()