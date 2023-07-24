import cv2
from math import *
import numpy as np
import time
import os
from tqdm import tqdm


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


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


def crop(img_path: str, pos: list, cls: str):
    img = cv2.imread(img_path)
    # print(cls)
    # points for test.jpg
    cnt = np.array([
        [[pos[0], pos[1]]],
        [[pos[2], pos[3]]],
        [[pos[4], pos[5]]],
        [[pos[6], pos[7]]]
    ], dtype=np.float32)
    # print("shape of cnt: {}".format(cnt.shape))
    rect = cv2.minAreaRect(cnt)
    # print("rect: {}".format(rect))

    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int64(box)

    # print("bounding box: {}".format(box))
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))

    return warped, cls


def drawRect(img, pt1, pt2, pt3, pt4, color, lineWidth):
    cv2.line(img, pt1, pt2, color, lineWidth)
    cv2.line(img, pt2, pt3, color, lineWidth)
    cv2.line(img, pt3, pt4, color, lineWidth)
    cv2.line(img, pt1, pt4, color, lineWidth)


if __name__ == '__main__':
    data_root = '../data/2023/train/'
    imgfiles = os.listdir(data_root + 'images')
    for imgfile in tqdm(imgfiles):
        img_path = os.path.join(data_root + 'images', imgfile)
        count = 0
        basename = get_file_basename(imgfile)
        label_txt = data_root + 'labels/' + basename + '.txt'
        with open(label_txt, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for l in lines:
                alist = []
                for i in range(0, 8):
                    alist.append(float(l.split(' ')[i]))
                cls = l.split(' ')[8].replace('\n', '')
                outimg = crop(img_path, alist, cls)
                count += 1
                outdir = './cls_images/' + cls
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                cv2.imwrite(outdir + '/' + basename + '_' + str(count) + '.jpg', outimg)

    # imgSrc = np.array(cv2.imread('run2_train_00007__1024__0___0.png'))
    # # imgSrc = np.zeros((1024, 1024, 3), np.float32)
    # print(imgSrc)
    # # pt1 = (int(357.8), int(619.0))
    # # pt2 = (int(379.1), int(641.0))
    # # pt3 = (int(220.0), int(794.5))
    # # pt4 = (int(198.7), int(772.5))
    # pt1, pt2, pt3, pt4 = (int(357.8), int(619.0)), (int(379.1), int(641.0)), (int(220.0), int(794.5)), (
    #     int(198.7), int(772.5))
    # imgSrc = cv2.line(imgSrc, pt1, pt2, (0, 0, 255), 2)
    # imgSrc = cv2.line(imgSrc, pt2, pt3, (0, 0, 255), 2)
    # imgSrc = cv2.line(imgSrc, pt3, pt4, (0, 0, 255), 2)
    # imgSrc = cv2.line(imgSrc, pt4, pt1, (0, 0, 255), 2)
    # # cv2.imshow('image', imgSrc)
    #
    # cv2.waitKey(0)
    # cv2.imwrite("imgRotation.jpg", imgSrc)
