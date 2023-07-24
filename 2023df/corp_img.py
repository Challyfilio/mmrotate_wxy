import cv2
from math import *
import numpy as np
import time


def polygonToRotRectangle_batch(bbox, with_module=True):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
            shape [num_boxes, 8]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
            shape [num_rot_recs, 5]
    """
    # print('bbox: ', bbox)
    bbox = np.array(bbox, dtype=np.float32)
    bbox = np.reshape(bbox, newshape=(-1, 2, 4), order='F')
    # angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    # print('bbox: ', bbox)
    angle = np.arctan2(-(bbox[:, 0, 1] - bbox[:, 0, 0]), bbox[:, 1, 1] - bbox[:, 1, 0])
    # angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # center = [[0],[0]] ## shape [2, 1]
    # print('angle: ', angle)
    center = np.zeros((bbox.shape[0], 2, 1))
    for i in range(4):
        center[:, 0, 0] += bbox[:, 0, i]
        center[:, 1, 0] += bbox[:, 1, i]

    center = np.array(center, dtype=np.float32) / 4.0

    # R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose((2, 1, 0)), bbox - center)

    xmin = np.min(normalized[:, 0, :], axis=1)
    # print('diff: ', (xmin - normalized[:, 0, 3]))
    # assert sum((abs(xmin - normalized[:, 0, 3])) > eps) == 0
    xmax = np.max(normalized[:, 0, :], axis=1)
    # assert sum(abs(xmax - normalized[:, 0, 1]) > eps) == 0
    # print('diff2: ', xmax - normalized[:, 0, 1])
    ymin = np.min(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymin - normalized[:, 1, 3]) > eps) == 0
    # print('diff3: ', ymin - normalized[:, 1, 3])
    ymax = np.max(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymax - normalized[:, 1, 1]) > eps) == 0
    # print('diff4: ', ymax - normalized[:, 1, 1])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    # TODO: check it
    if with_module:
        angle = angle[:, np.newaxis] % (2 * np.pi)
    else:
        angle = angle[:, np.newaxis]
    dboxes = np.concatenate((center[:, 0].astype(float), center[:, 1].astype(float), w, h, angle), axis=1)
    return dboxes  # [cx, cy, w, h, theta]


def rotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)
    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgOut = imgRotation[int(pt1[1]):int(pt3[1]), int(pt1[0]):int(pt3[0])]
    cv2.imshow("imgOut", imgOut)  # 裁减得到的旋转矩形框
    cv2.imwrite("imgOut.jpg", imgOut)
    # pt2 = list(pt2)
    # pt4 = list(pt4)
    # [[pt2[0]], [pt2[1]]] = np.dot(matRotation, np.array([[pt2[0]], [pt2[1]], [1]]))
    # [[pt4[0]], [pt4[1]]] = np.dot(matRotation, np.array([[pt4[0]], [pt4[1]], [1]]))
    # pt1 = (int(pt1[0]), int(pt1[1]))
    # pt2 = (int(pt2[0]), int(pt2[1]))
    # pt3 = (int(pt3[0]), int(pt3[1]))
    # pt4 = (int(pt4[0]), int(pt4[1]))
    # drawRect(imgRotation,pt1,pt2,pt3,pt4,(255,0,0),2)
    return imgRotation


def drawRect(img, pt1, pt2, pt3, pt4, color, lineWidth):
    cv2.line(img, pt1, pt2, color, lineWidth)
    cv2.line(img, pt2, pt3, color, lineWidth)
    cv2.line(img, pt3, pt4, color, lineWidth)
    cv2.line(img, pt1, pt4, color, lineWidth)


if __name__ == '__main__':
    # bbox = [357.8, 619.0, 379.1, 641.0, 220.0, 794.5, 198.7, 772.5]
    # print(polygonToRotRectangle_batch(bbox))
    #
    # startTime = time.time()
    # imgSrc = cv2.imread('run2_train_00007__1024__0___0.png')
    # # imgResize = cv2.resize(imgSrc, (500, 500))
    # pt1 = (357.8, 619.0)
    # pt2 = (379.1, 641.0)
    # pt3 = (220.0, 794.5)
    # pt4 = (198.7, 772.5)
    # # drawRect(imgResize,pt1,pt2,pt3,pt4,(0,0,255),2)
    # imgRotation = rotateImage(imgSrc, -degrees(atan2(50, 50)), pt1, pt2, pt3, pt4)
    # endTime = time.time()
    # print(endTime - startTime)
    # cv2.imshow("imgRotation", imgRotation)
    # cv2.imwrite("imgRotation.jpg", imgRotation)
    # cv2.waitKey(0)

    imgSrc = np.array(cv2.imread('run2_train_00007__1024__0___0.png'))
    # imgSrc = np.zeros((1024, 1024, 3), np.float32)
    print(imgSrc)
    # pt1 = (int(357.8), int(619.0))
    # pt2 = (int(379.1), int(641.0))
    # pt3 = (int(220.0), int(794.5))
    # pt4 = (int(198.7), int(772.5))
    pt1, pt2, pt3, pt4 = (int(357.8), int(619.0)), (int(379.1), int(641.0)), (int(220.0), int(794.5)), (int(198.7), int(772.5))
    imgSrc = cv2.line(imgSrc, pt1, pt2, (0, 0, 255), 2)
    imgSrc = cv2.line(imgSrc, pt2, pt3, (0, 0, 255), 2)
    imgSrc = cv2.line(imgSrc, pt3, pt4, (0, 0, 255), 2)
    imgSrc = cv2.line(imgSrc, pt4, pt1, (0, 0, 255), 2)
    # cv2.imshow('image', imgSrc)

    cv2.waitKey(0)
    cv2.imwrite("imgRotation.jpg", imgSrc)
