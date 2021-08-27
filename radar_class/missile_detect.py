'''
飞镖预警类
飞镖预警调过的参数直接放在函数里，不放在整体的config
'''
import cv2
import numpy as np
import traceback
import time

from radar_class.common import is_inside
from radar_class.config import enemy2color,two_stage_time


def missile_filter(frame_m, red=True):
    ######## param #############
    # H域的上下限
    if red:
        l = 170
        h = 180
    else:
        l = 95
        h = 100
    intensity_thre = 175
    #########################
    # intensity threshold
    gray = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    gray[gray < intensity_thre] = 0
    # hsv threshold
    frame_hsv = cv2.cvtColor(frame_m, cv2.COLOR_BGR2HSV)
    frame_ii = cv2.inRange(frame_hsv, (l, 170, 0), (h, 255, 255))
    # to dilate the small area
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    frame_ii = cv2.dilate(frame_ii, kernel)
    gray[frame_ii < 200] = 0  # reserve the responding area
    return gray


class Missile(object):
    '''
    飞镖预警类
    '''
    # 下列参数均为第一阶段参数
    _region_thre = [100, 40000]  # 响应区域bounding box大小上下限
    _intensity_bound = 175

    def __init__(self, touch_api, enemy, debug=False):
        '''
        missile detector

        :param touch_api:与裁判系统通信api 格式： f(message:dict)
        :param enemy: enemy number (red is 0 blue is 1)
        :param debug:using debug mode you can adjust the thresholds of
        both the area of the bounding box of the corners and the intensity to filter the image
        '''
        self._init_flag = False
        self._touch_api = touch_api
        self._debug = debug
        self._enemy = enemy
        self._two_stage_time = 0  # 第二阶段开始的时间
        self._region = [None, None]
        self._roi = [None, None]
        self._previous_frame = [None, None]
        if debug:
            cv2.namedWindow("missile_debug", cv2.WINDOW_NORMAL)
            cv2.createTrackbar("lb", "missile_debug", 0, 1, lambda x: None)
            cv2.setTrackbarMax("lb", "missile_debug", 1000)
            cv2.setTrackbarMin("lb", "missile_debug", 100)
            cv2.setTrackbarPos("lb", "missile_debug", 500)
            cv2.createTrackbar("s", "missile_debug", 0, 1, lambda x: None)
            cv2.setTrackbarMax("s", "missile_debug", 255)
            cv2.setTrackbarMin("s", "missile_debug", 0)
            cv2.setTrackbarPos("s", "missile_debug", 200)

    def _push_init_project(self, init_frame, region: dict):
        # to compute and cache the region(four points to describe) for detecting, if this object haven't been initialized
        if not self._init_flag:
            # two stages use different region to check
            for i in range(2):
                r = region[f's_fp_{enemy2color[self._enemy]}_missilelaunch{i + 1:d}_d'].copy()  # 凸四边形

                rect = cv2.boundingRect(r)
                r -= np.array(rect[:2])  # 以外bounding box左上角为原点的凸四边形坐标

                # 存储取图像ROI区域的信息
                self._roi[i] = rect
                self._region[i] = r.copy()
                # 存储帧差第一帧
                self._previous_frame[i] = init_frame[self._roi[i][1]:(self._roi[i][1] + self._roi[i][3]),
                                          self._roi[i][0]:(self._roi[i][0] + self._roi[i][2])].copy()
            self._init_flag = True

    def init_two_stage(self):
        '''
        云台手按下按钮后，启动计时
        '''
        self._two_stage_time = time.time()

    def detect_two_stage(self, img, region=None):
        '''

        '''
        launch = False
        detect_flag = self.detect(img, region, 1)
        if time.time() - self._two_stage_time > two_stage_time:  # 若处于反导侦测阶段TWO_STAGE_TIME秒，则自动结束
            print("missile end")
            return False, False
        if detect_flag:
            self._touch_api({"task": 4})
            launch = True
        return True, launch

    def detect(self, img: np.ndarray, region: dict, stage):
        '''
        二值化+帧差法检测指示灯&飞镖

        可作为第一阶段检测被他类调用，也作为第二阶段检测函数被本类调用

        :param img:当前帧输入
        :param region:全部的反投影预警区域输入后会自动筛取对应区域，仅在初始化时使用
        :param stage:飞镖预警阶段 0 is the first stage 1 is the second stage
        '''
        if self._debug:
            self._region_thre[0] = cv2.getTrackbarPos("lb", "missile_debug")
            self._intensity_bound = cv2.getTrackbarPos("s", "missile_debug")
        try:
            # 若未初始化
            if not self._init_flag:
                self._push_init_project(img, region)
                return False
            else:
                detect_flag = False
                # 取对应阶段当前帧ROI区域
                current_frame = img[self._roi[stage][1]:(self._roi[stage][1] + self._roi[stage][3]),
                                self._roi[stage][0]:(self._roi[stage][0] + self._roi[stage][2])].copy()
                if stage == 0:
                    # 第一阶段处理，亮度二值化+帧差
                    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                    previous_frame_gray = cv2.cvtColor(self._previous_frame[stage], cv2.COLOR_BGR2GRAY)
                    for i in range(2):
                        # cache the previous frame of both the two stage
                        self._previous_frame[i] = img[self._roi[i][1]:(self._roi[i][1] + self._roi[i][3]),
                                                  self._roi[i][0]:(self._roi[i][0] + self._roi[i][2])].copy()
                    current_frame_gray = cv2.GaussianBlur(current_frame_gray, (7, 7), 0)
                    previous_frame_gray = cv2.GaussianBlur(previous_frame_gray, (7, 7), 0)
                    current_frame_gray[current_frame_gray < self._intensity_bound] = 0
                    previous_frame_gray[previous_frame_gray < self._intensity_bound] = 0
                    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
                else:
                    # 第二阶段处理， 飞镖滤波函数（亮度+HSV区域二值化） + 帧差
                    current_frame_filter = missile_filter(current_frame,
                                                          not self._enemy)  # not self._enemy 来选择是进行红滤波还是蓝滤波

                    previous_frame_filter = missile_filter(self._previous_frame[stage], not self._enemy)
                    for i in range(2):
                        # cache the previous frame of both the two stage
                        self._previous_frame[i] = img[self._roi[i][1]:(self._roi[i][1] + self._roi[i][3]),
                                                  self._roi[i][0]:(self._roi[i][0] + self._roi[i][2])].copy()
                    frame_diff = cv2.absdiff(current_frame_filter, previous_frame_filter)
                if stage == 0:
                    # stage 1进行差值二值化，stage 2则不需要因为差值要么是0要么是255
                    _, frame_diff = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

                # stage 2阈值调得十分窄，故而滤出来区域很小，只进行dilate来连通其区域
                if stage == 0:
                    frame_diff = cv2.erode(frame_diff, kernel)
                frame_diff = cv2.dilate(frame_diff, kernel)

                contours, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if stage == 0:
                    for c in contours:
                        if self._region_thre[0] < cv2.contourArea(c) < self._region_thre[1]:
                            x, y, w, h = cv2.boundingRect(c)
                            # 中心点是否在凸四边形区域内
                            flag = is_inside(self._region[stage], np.array([x + w // 2, y + h // 2]))
                            if flag:
                                detect_flag = True
                                print("First stage detect")
                                if self._debug:
                                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 0, 255))
                                else:
                                    break
                else:
                    for c in contours:
                        if cv2.contourArea(c) > 30:  # 轮廓面积阈值
                            x, y, w, h = cv2.boundingRect(c)
                            flag = is_inside(self._region[stage], np.array([x + w // 2, y + h // 2]))
                            if flag:
                                print("Second stage detect")
                                detect_flag = True
                                if self._debug:
                                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 0, 255))
                                else:
                                    break

                if self._debug:
                    cv2.imshow('missile_debug', current_frame)
                return detect_flag
        except Exception:
            traceback.print_exc()
            return False

    def __del__(self):
        if self._debug:
            cv2.destroyWindow("missile_debug")
