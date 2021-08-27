'''
common.py
对于所有类都有用的函数
'''
import numpy as np
import cv2

from radar_class.config import color2enemy

def is_inside(box: np.ndarray, point: np.ndarray):
    '''
    判断点是否在凸四边形中

    :param box:为凸四边形的四点 shape is (4,2)
    :param point:为需判断的是否在内的点 shape is (2,)
    '''
    assert box.shape == (4, 2)
    assert point.shape == (2,)
    AM = point - box[0]
    AB = box[1] - box[0]
    BM = point - box[1]
    BC = box[2] - box[1]
    CM = point - box[2]
    CD = box[3] - box[2]
    DM = point - box[3]
    DA = box[0] - box[3]
    a = np.cross(AM, AB)
    b = np.cross(BM, BC)
    c = np.cross(CM, CD)
    d = np.cross(DM, DA)
    return a >= 0 and b >= 0 and c >= 0 and d >= 0 or \
           a <= 0 and b <= 0 and c <= 0 and d <= 0


def armor_plot(location, img):
    '''
    画四点装甲板框

    :param location: np.ndarray (N,armor_points + data) 即后一个维度前八位必须是按顺时针顺序的装甲板四点坐标
    '''
    if isinstance(location, np.ndarray):
        for gl in location:
            l = gl[:8].reshape(4, 2).astype(np.int)
            for i in range(len(l)):
                cv2.line(img, tuple(l[i]), tuple(l[(i + 1) % len(l)]), (0, 255, 0), 2)


def plot(results, frame, only_car=True):
    '''
    画车辆预测框

    :param results: list, every item is (predicted_class,conf_score,bounding box(format:x0,y0,x1,y1))
    :param frame: the image to plot on it
    :param only_car:ignore the watcher(guard) and base class

    :return: 当输入有仅有颜色预测框时，返回该类预测框的bbox和其他对应id的bbox整合
    '''
    color_bbox = []
    # plot on the raw frame
    for cat, score, bound in results:
        if cat in "watcher base" and only_car:
            continue
        if '0' in cat:
            # 通过颜色检测出的bounding box
            color_bbox.append(np.array([color2enemy[cat.split('_')[1]], *bound]))
        plot_one_box(cat, bound, frame)
    if len(color_bbox):
        return np.stack(color_bbox, axis=0)
    else:
        return None


def plot_one_box(cat, b, img):
    if cat.split('_')[0] == "car":
        if len(cat.split('_')) == 3:  # car_{armor_color}_{armor_id}
            cat = cat.split('_')
            color = cat[1]
            armor = cat[2]
        else:
            color = "C"
            armor = '0'
    else:
        # 只预测出颜色情况
        color = cat  # ‘R0’ or 'B0'
        armor = '0'
    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    cv2.putText(img, color[0].upper() + armor, (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_SIMPLEX,
                3 * img.shape[1] // 3088, (255, 0, 255), 2)


def armor_filter(armors):
    '''
    装甲板去重

    :param armors:input np.ndarray (N,fp+conf+cls+img_no+bbox)

    :return: armors np.ndarray 每个id都最多有一个装甲板
    '''

    # 直接取最高置信度
    ids = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12] # 1-5分别为b1-5 8-12分别为r1-5
    if isinstance(armors, np.ndarray):
        results = []
        for i in ids:
            mask = armors[:, 9] == i
            armors_mask = armors[mask]
            if armors_mask.shape[0]:
                armor = armors_mask[np.argmax(armors_mask[:, 8])]
                results.append(armor)
        if len(results):
            armors = np.stack(results, axis=0)
            return armors
        else:
            return None
    else:
        return None


def car_classify(frame_m, red=True):
    '''
    亮度阈值加HSV判断车辆颜色

    :param frame_m:输入图像（可以是ROI)
    :param red:判断为红还是蓝

    :return: 判断结果
    '''
    ########param#############
    if red:
        l = 10
        h = 30
    else:
        l = 88
        h = 128
    intensity_thre = 200
    channel_thre = 150
    #########################
    frame_ii = np.zeros((frame_m.shape[0], frame_m.shape[1]), dtype=np.uint8)
    # intensity threshold
    gray = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    mask_intensity = gray > intensity_thre
    frame_hsv = cv2.cvtColor(frame_m, cv2.COLOR_BGR2HSV)
    mask = np.logical_and(frame_hsv[:, :, 0] < h, frame_hsv[:, :, 0] > l)
    b, g, r = cv2.split(frame_m)
    # 通道差阈值过滤
    if red:
        mask_color = (r - b) > channel_thre
    else:
        mask_color = (b - r) > channel_thre
    frame_ii[np.logical_and(np.logical_and(mask, mask_color), mask_intensity)] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    frame_ii = cv2.dilate(frame_ii, kernel)
    gray[frame_ii < 200] = 0
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flag = False
    for c in contours:
        if cv2.contourArea(c) > 5:
            flag = True
    return flag
