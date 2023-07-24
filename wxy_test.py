import torch
import torch.nn.functional as F
import torch
from torch import nn

from mmdet.models.necks.fpn import FPN
from mmrotate.models.necks.spfn import SFPN
from mmrotate.models.necks.cslfpn import CSLFPN

from loguru import logger

if __name__ == '__main__':
    input_tensor2 = torch.rand(2, 96, 256, 256)
    input_tensor3 = torch.rand(2, 192, 128, 128)
    input_tensor4 = torch.rand(2, 384, 64, 64)
    input_tensor5 = torch.rand(2, 768, 32, 32)
    channels50 = [256, 512, 1024, 2048] #rn50
    channels34 = [64, 128, 256, 512] #rn34
    channels_s = [96, 192, 384, 768]
    model = CSLFPN(feature_map_shape=[256,128,64,32],
                   in_channels=channels_s,
                   out_channels=256,
                   num_outs=5)
    # model = SFPN(in_channels=channels_s,
    #              out_channels=256,
    #              num_outs=5)
    outputs = model((input_tensor2,input_tensor3,input_tensor4,input_tensor5))
    logger.success(len(outputs))
    for j in outputs:
        print(j.shape)
    exit()

    alpha = 0.25
    gamma = 2
    epsilon = 0.9
    # pred = torch.randn(2,3)
    pred = torch.tensor([[0.8,0.3],[0.4,0.6]])
    target = torch.tensor([1,0])
    print(pred)
    print(target)
    pt = F.log_softmax(pred, dim=-1) # pt
    print(pt)
    nll_1 = F.nll_loss(pt, target, reduction='mean')
    nll_2 = F.nll_loss(1-pt, target, reduction='mean')
    # print(log_preds)
    # print(1-log_preds)
    loss = alpha * (1 - pt) ** gamma * epsilon * nll_1 + (1 - alpha) * pt ** gamma * (1 - epsilon) * nll_2
    print(loss)

    # #随机生成一个神经网络的最后一层，3行4列，那就是有4个标签
    # input = torch.randn(3,4)
    # #input的第一行设置为标签1，第二行为标签0,
    # label = torch.tensor([1,0,2])

    # #人工设置种子，不然每次计算loss不一样，我们通过固定种子就可以固定loss
    # torch.manual_seed(2)

    # #定义损失函数为NLLLoss
    # loss = nn.NLLLoss()
    # #定义log softmax函数，也就是将input中的每一行转化为带有负号的数字
    # m = nn.LogSoftmax(dim=1)
    # #计算损失，损失就是一个值。
    # loss_value = loss(m(input),label)
    # print(loss_value)