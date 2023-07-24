import os
import json

import torch
from PIL import Image
from torchvision import transforms, models

#from model import resnet50


def predict_single(img):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # img_path = "/workspace/pycharm_project/Columba-main/custom_dataset_cls/val/A7/run2_train_01730_7.jpg"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    # plt.imshow(img)
    # [N, C, H, W]
    img = Image.fromarray(img)
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = models.resnet50(pretrained=False, num_classes=98).to(device)
#    model = resnet50(num_classes=98).to(device)

    # load model weights
    weights_path = "./best_model.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        # print(output)
        predict = torch.softmax(output, dim=0)
        # print(predict)
        predict_cla = torch.argmax(predict).numpy()
        # print(predict[predict_cla].numpy())
        # print(predict_cla)
        # print(class_indict[str(predict_cla)])

    return class_indict[str(predict_cla)], predict[predict_cla].numpy()
    # return predict_cla


if __name__ == '__main__':
    pass
