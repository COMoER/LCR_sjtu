'''
这是整个雷达站的主程序，里面有一些全局参数，是要在比赛前修改的，请在radar_class/config.py内设置
'''
import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets
from datetime import datetime
import time
import os
import multiprocessing as mlt
import pickle as pkl

# 各个对象
from radar_class.Lidar import Radar,DepthQueue
from mainEntry import Mywindow
from radar_class.camera import Camera_Thread, read_yaml
from radar_class.reproject import Real_Scene
from radar_class.location_alarm import Alarm
from radar_class.location import locate_pick, locate_record
from radar_class.config import enemy,usb, \
    color2enemy,loc2car,loc2code,state2color,\
    INIT_FRAME_PATH,OUTPUT_LOGGING_DIR,SPECIFIC_LOGGING_BATTLE_MODE_DIR,SPECIFIC_LOGGING_NON_BATTLE_MODE_DIR,DEMO_PKL_DIR,\
    main_region,battle_mode,real_size,using_dq,PC_RECORD_SAVE_DIR,debug,home_test,using_video,VIDEO_PATH,split # 测试参
from radar_class.common import armor_filter
from radar_class.multiprocess_camera import base_process
from radar_class.ui import HP_scene
# TODO: 替换你的神经网络
from radar_class.network import Predictor
# 串口库函数
import serial
import _thread
import pexpect

from UART import read, write, UART_passer

record_net = []

# 裁判系统发送函数
def send_judge(message: dict, myshow: Mywindow, f):
    '''
    alarming format define
    {'task': 2, 'data': [team, loc, targets]}
    {'task': 3, 'data': [targets]}

    ps: targets : a list of the car inside the region
    '''
    if message['task'] == 1:  # map location send
        whole_location: dict = message['data'][0]
        if myshow.record_state and battle_mode:
            for label, cl in whole_location.items():
                # logging记录
                f.write("target %s x y z: %.03f %.03f %.03f\n"% (label, cl[0], cl[1], cl[2]))
        loc = []
        whole_location: dict = message['data'][1]
        for i in range(1, 6):
            if home_test:
                # 裁判系统里定义的为(28.,15.),等比例放大
                whole_location[str(i + enemy * 5)][0] *= 28. / real_size[0]
                whole_location[str(i + enemy * 5)][1] *= 15. / real_size[1]
            loc.append(whole_location[str(i + enemy * 5)][:2])

        UART_passer.push_loc(loc)

    if message['task'] == 2:  # alarming
        team, loc, targets = message['data']
        UART_passer.push(loc2code[loc], loc2car[loc], targets, color2enemy[team])
    if message['task'] == 3:  # base alarming
        targets = message['data']
        UART_passer.push(loc2code['base'], loc2car['base'], targets, 0)
    if message['task'] == 4: # anti dart
        UART_passer.push(loc2code['dart'], loc2car['dart'], [], 0)
        myshow.set_text("feedback", "<font color='#FF0000'><b>反导信息发送</b></font>")


# 主程序类
class radar_process(object):
    def __init__(self, myshow: Mywindow, log_f):
        self._log_f = log_f # logging
        self._myshow = myshow
        # 展示api定义
        self._show_api_m = lambda x: myshow.set_image(x, "map")
        self._show_api_s = lambda x: myshow.set_image(x, "main_demo")
        self._touch_api = lambda x: send_judge(x, myshow, log_f)
        self._radar = []
        self._scene = []
        frame = cv2.imread(INIT_FRAME_PATH)  # init frame of scene
        # get position info
        # camera0 Radar class and Real_Scene class
        _, K_0, C_0, E_0, imgsz = read_yaml(0)
        if using_dq: # 使用点云队列
            self._radar.append(DepthQueue(200,imgsz,K_0,C_0,E_0))
            # 填入点云
            with open(PC_RECORD_SAVE_DIR, 'rb') as f:
                radar_frames = pkl.load(f)
            for frame_r in radar_frames:
                self._radar[0].push_back(frame_r)
        else:
            self._radar.append(Radar(K_0, C_0, E_0, imgsz=imgsz))
        self._scene.append(Real_Scene(frame, 0, main_region, enemy, real_size, K_0, C_0, self._touch_api, debug = debug))
        # camera0 Radar class and scene class
        _, K_0, C_0, E_0, imgsz = read_yaml(1)
        if using_dq: # 使用点云队列
            self._radar.append(DepthQueue(200,imgsz,K_0,C_0,E_0))
            # 填入点云
            with open(PC_RECORD_SAVE_DIR, 'rb') as f:
                radar_frames = pkl.load(f)
            for frame_r in radar_frames:
                self._radar[1].push_back(frame_r)
        else:
            self._radar.append(Radar(K_0, C_0, E_0, imgsz=imgsz))
        self._scene.append(Real_Scene(frame, 1, main_region, enemy, real_size, K_0, C_0, self._touch_api, debug = debug))
        # 位姿估计均未完成
        self._position_flag = np.array([False, False])
        # 位置预警类初始化
        self._alarm_map = Alarm(main_region, self._show_api_m, self._touch_api, enemy, real_size,debug=debug)
        if not using_dq:
            Radar.start() # 雷达开始工作
        # base process init
        self._loc2basequeue = mlt.Queue(-1)
        self._manager = mlt.Manager()
        self._loc2baseflag = self._manager.dict({"flag": False, 'record': False, 'battle_mode':battle_mode})
        self._base_process = mlt.Process(target=base_process, args=(self._loc2basequeue, self._loc2baseflag),daemon=True)

        self._base_process.start()
        # 雷达队列初始化标志
        if using_dq:
            self._radar_init = [True,True] # 当使用点云队列测试时，直接默认为双True
        else:
            self._radar_init = [False,False]

        # init net #TODO: 替换你的神经网络
        self._net = Predictor(DEMO_PKL_DIR)
        print("[INFO] net init finish")

        # open main process camera
        if using_video:
            self._cap1 = Camera_Thread(0, True,
                                       video_path=os.path.join(VIDEO_PATH,"1.mp4"))
            self._cap2 = Camera_Thread(1, True,
                                       video_path=os.path.join(VIDEO_PATH,"2.mp4"))
        else:
            self._cap1 = Camera_Thread(0)
            self._cap2 = Camera_Thread(1)

        if self._cap1.is_open():
            print("[INFO] Camera {0} Starting.".format(0))
        else:
            print("[INFO] Camera {0} Failed, try to open.".format(0))
        if self._cap2.is_open():
            print("[INFO] Camera {0} Starting.".format(1))
        else:
            print("[INFO] Camera {0} Failed, try to open.".format(1))

        # recording counting
        if battle_mode:
            self._c1_frame = 0
            self._c2_frame = 0

    def join(self):
        '''
        关闭子进程，使其不要成为僵尸进程（zombie process）
        '''
        self._base_process.join()
        self._base_process.close()

    def get_position_using_last(self):
        '''
        使用保存的位姿
        '''
        if self._position_flag.all(): # 全部位姿已估计完成
            self._myshow.set_text("feedback", "camera pose already init")
            return
        if not self._position_flag[0]:
            flag, rvec1, tvec1 = locate_record(0, enemy)
            if flag:
                self._position_flag[0] = True
                myshow.set_text("feedback", "Camera 0 pose init")
                print("[INFO] Camera 0 pose init")
                # 将位姿存入反投影预警类
                T, cp = self._scene[0].push_T(rvec1, tvec1)
                # 将位姿存入位置预警类
                self._alarm_map.push_T( T, cp, 0)
            else:
                myshow.set_text("feedback", "Camera 0 pose init error")
                print("[INFO] Camera 0 pose init meet error")
        if not self._position_flag[1]:
            flag, rvec2, tvec2 = locate_record(1, enemy)
            if flag:
                self._position_flag[1] = True
                self._myshow.set_text("feedback", "camera 1 pose init ")
                print("[INFO] Camera 1 pose init")
                T, cp = self._scene[1].push_T(rvec2, tvec2)
                self._alarm_map.push_T( T, cp, 1)
            else:
                self._myshow.set_text("feedback", "camera 1 pose init meet error ")
                print("[INFO] Camera 1 pose init meet error")
        # 重叠区域去重
        if split and self._position_flag.all():
            inside1, whole_location = self._scene[0].get_inside()
            inside1 = np.stack(inside1, axis=0)
            inside2, _ = self._scene[1].get_inside()
            inside2 = np.stack(inside2, axis=0)
            both_inside = np.logical_and(inside1.any(axis=1), inside2.any(axis=1)) # 大家都有的区域
            larger = inside1.sum(axis=1) >= inside2.sum(axis=1) # 比较在图像内的点数
            lesser = np.logical_not(larger)
            # larger为c0比c1多，lesser为c1比c0多
            whole_location = np.array(whole_location)
            self._scene[0].remove(whole_location[np.logical_and(lesser,both_inside)])
            self._scene[1].remove(whole_location[np.logical_and(both_inside,larger)])

    def get_position_new(self):
        '''
        using huge range object to get position, which is simple but perfect
        '''
        if self._position_flag.all():
            self._myshow.set_text("feedback", "camera pose already init")
            return

        if not self._position_flag[0]:
            flag = False
            if self._cap1.is_open():
                flag, rvec1, tvec1 = locate_pick(self._cap1, enemy, 0,  home_size= home_test, video_test=using_video)
            if flag:
                self._position_flag[0] = True
                print("[INFO] Camera 0 pose init")
                myshow.set_text("feedback", "Camera 0 pose init")
                if battle_mode:  # do logging
                    camera_type = 0
                    self._log_f.write("transform{0}:\n".format(camera_type))
                    self._log_f.write(
                        f"camera{camera_type:d} rvec {float(rvec1[0]):0.5f} {float(rvec1[1]):0.5f} {float(rvec1[2]):0.5f}\n")
                    self._log_f.write(
                        f"camera{camera_type:d} tvec {float(tvec1[0]):0.5f} {float(tvec1[1]):0.5f} {float(tvec1[2]):0.5f}\n")
                locate_record(0, enemy, True, rvec1, tvec1) # 保存
                T, cp = self._scene[0].push_T(rvec1, tvec1)
                self._alarm_map.push_T( T, cp, 0)
            else:
                myshow.set_text("feedback", "Camera 0 pose init meet error")
                print("[INFO] Camera 0 pose init error")

        if not self._position_flag[1]:
            flag = False
            if self._cap2.is_open():
                flag, rvec2, tvec2 = locate_pick(self._cap2, enemy, 1, home_size = home_test, video_test=using_video)
            if flag:
                self._position_flag[1] = True
                if battle_mode:
                    camera_type = 1
                    self._log_f.write("transform{0}:\n".format(camera_type))

                    self._log_f.write(
                        f"camera{camera_type:d} rvec {float(rvec2[0]):0.5f} {float(rvec2[1]):0.5f} {float(rvec2[2]):0.5f}\n")
                    self._log_f.write(
                        f"camera{camera_type:d} tvec {float(tvec2[0]):0.5f} {float(tvec2[1]):0.5f} {float(tvec2[2]):0.5f}\n")
                locate_record(1, enemy, True, rvec2, tvec2)
                T, cp = self._scene[1].push_T(rvec2, tvec2)
                self._alarm_map.push_T(T, cp, 1)
                print("[INFO] Camera 1 pose init")
                self._myshow.set_text("feedback", "camera 1 pose init ")
            else:
                self._myshow.set_text("feedback", "camera 1 pose init meet error")
                print("[INFO] Camera 1 pose init error")
        # 如果使用视频，将其恢复至视频开始
        if using_video:
            self._cap1.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._cap2.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # split
        if split and self._position_flag.all():
            inside1, whole_location = self._scene[0].get_inside()
            inside1 = np.stack(inside1, axis=0)
            inside2, _ = self._scene[1].get_inside()
            inside2 = np.stack(inside2, axis=0)
            both_inside = np.logical_and(inside1.any(axis=1), inside2.any(axis=1))
            larger = inside1.sum(axis=1) >= inside2.sum(axis=1)
            lesser = np.logical_not(larger)
            whole_location = np.array(whole_location)
            self._scene[0].remove(whole_location[np.logical_and(lesser, both_inside)])
            self._scene[1].remove(whole_location[np.logical_and(both_inside, larger)])

    def spin_once(self, base_open=False, base_close=False, whole_close=False, is_opening=False):
        '''
        雷达站主程序的一个循环

        :param base_open:开启子视野
        :param base_close:关闭子视野
        :param whole_close:程序终止
        :param is_opening:代表已经子进程相机视野开启的命令
        '''
        # check radar init
        if not using_dq:
            if self._radar[0].check_radar_init():
                self._myshow.set_text("feedback", "radar of camera 0 init")
                self._radar_init[0] = True
                print("[INFO] radar of camera 0 init")
            if self._radar[1].check_radar_init():
                self._radar_init[1] = True
                self._myshow.set_text("feedback", "radar of camera 1 init")
                print("[INFO] radar of camera 1 init")

        if whole_close:
            # close the program
            self._loc2basequeue.put(2)
            return
        if base_open:
            self._loc2basequeue.put(1) # task1
        if base_close:
            self._loc2basequeue.put(0) # task0

        if self._myshow.record_state and battle_mode and not self._loc2baseflag["record"]:
            self._loc2baseflag["record"] = True # 打开子视野的录制模型

        if not self._myshow.record_state and self._loc2baseflag["record"]:
            self._loc2baseflag["record"] = False # 关闭子视野的录制模型

        if is_opening and not self._loc2baseflag['flag']:
            # 当要求开启但未开启时，报错（一般一定会报一次，这是因为进程同步问题）
            self._myshow.set_text("feedback", "CAMERA3 may meet some problem please waiting")

        # show state for check
        self._myshow.set_text("state", '<br \>'.join(["Radar1: " + "<font color='{0}'><b>{1}</b></font>".format(
            state2color[self._radar_init[0]], self._radar_init[0]),
                                                      "Radar2: " + "<font color='{0}'><b>{1}</b></font>".format(
                                                          state2color[self._radar_init[1]], self._radar_init[1]),
                                                      "Pose1: " + "<font color='{0}'><b>{1}</b></font>".format(
                                                          state2color[self._position_flag[0]], self._position_flag[0]),
                                                      "Pose2: " + "<font color='{0}'><b>{1}</b></font>".format(
                                                          state2color[self._position_flag[1]], self._position_flag[1])]
                                                     ))

        # get image and do prediction

        # if failed, try to reopen
        if not self._cap1.is_open():
            self._cap1.open()
        if not self._cap2.is_open():
            self._cap2.open()

        flag1, frame1 = self._cap1.read()
        flag2, frame2 = self._cap2.read()
        if not flag1 and not flag2:
            # 两个相机都崩了
            self._myshow.set_text("feedback", "ALL CAMERA BROKEN waiting until they resume")
            time.sleep(0.05)
            return
        imgs = []
        if flag1:
            imgs.append(frame1)
            # do recording
            if myshow.record_state and battle_mode:
                if battle_mode:
                    f.write(f"camera1 frame {self._c1_frame:d}\n")
                self._c1_frame += 1
                myshow.record_object[0].write(frame1)
        if flag2:
            imgs.append(frame2)
            # do recording
            if myshow.record_state and battle_mode:
                if battle_mode:
                    f.write(f"camera2 frame {self._c2_frame:d}\n")
                self._c2_frame += 1
                myshow.record_object[1].write(frame2)


        # 这里加入id是为了提取pkl列表的特定项,get可以获得视频当前位置，当你更换你的神经网络时应改为注释项
        # TODO: 替换你的神经网络
        results, location = self._net.infer(imgs,int(self._cap1.cap.get(cv2.CAP_PROP_POS_FRAMES))-1)
        # results, location = self._net.infer(imgs)

        # 以下均通过None先对某一预测结果对象进行填充，然后在得到预测结果时，将None改为np.ndarray，后面用类型判断来得到是否有预测
        pred1 = None
        pred2 = None

        # if only one camera working
        if len(imgs) == 1:
            location[0][0] = armor_filter(location[0][0])  # 填入全部滤过的装甲板
            if flag1:
                pred1 = [imgs[0], results[0], location[0]]
            if flag2:
                pred2 = [imgs[0], results[0], location[0]]
        if len(imgs) == 2:
            location[0][0] = armor_filter(location[0][0])  # 填入全部滤过的装甲板
            location[1][0] = armor_filter(location[1][0])  # 填入全部滤过的装甲板
            pred1 = [imgs[0], results[0], location[0]]
            pred2 = [imgs[1], results[1], location[1]]
        # update scene
        if isinstance(pred1, list):
            # check whether to start anti-dart
            if UART_passer.anti_dart:
                self._scene[0].open_missile_two_stage()
                self._myshow.set_text("feedback", "Start anti_missile two_stage")
                UART_passer.anti_dart = False
            self._scene[0].update(pred1[0], pred1[1], pred1[2][0])
        else:
            self._myshow.set_text("feedback", "CAMERA1 BROKEN the scene of it will be the same until it resume")
        if isinstance(pred2, list):
            self._scene[1].update(pred2[0], pred2[1], pred2[2][0])
        else:
            self._myshow.set_text("feedback", "CAMERA2 BROKEN the scene of it will be the same until it resume")
        # update location alarming class
        if self._position_flag.any():
            locations = [] # 直接神经网络预测装甲板位置
            extra_locations = [] # IoU预测位置
            if self._position_flag[0] and isinstance(pred1, list):
                _, extra_bbox = self._scene[0].check(pred1[2][0], pred1[2][1]) # 反投影预警检测+IoU预测
                locations.append(pred1[2][0])
                extra_locations.append(extra_bbox)
            else:
                locations.append(None)
                extra_locations.append(None)
            if self._position_flag[1] and isinstance(pred2, list):
                _, extra_bbox = self._scene[1].check(pred2[2][0], pred2[2][1])
                locations.append(pred2[2][0])
                extra_locations.append(extra_bbox)
            else:
                locations.append(None)
                extra_locations.append(None)

            self._alarm_map.two_camera_merge_update(locations, extra_locations, self._radar) # 合并预测

            na, ba = self._alarm_map.check() # na为是否有位置预警，ba为是否有基地预警

            self._scene[0].plot_alarming(na, ba)
            self._scene[1].plot_alarming(na, ba)

            # show updated map
            self._alarm_map.refresh()
            self._alarm_map.show()

        # 视角切换展示哪个相机输出
        if self._myshow.view_change:
            missile_alarm = self._scene[0].show_no_seen() # 接收右相机是否有飞镖预警
            self._scene[1].show(self._show_api_s,missile_alarm)

        else:
            missile_alarm = self._scene[0].show(self._show_api_s)
            self._scene[1].show_no_seen(missile_alarm)


#####logging###### 基于stdout和stderr来记录程序在运行过程中的输出
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

    def __del__(self):
        self.log.flush()
        self.log.close()


class Logger_Error(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stderr
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

    def __del__(self):
        self.log.flush()
        self.log.close()


###################


if __name__ == "__main__":
    import traceback

    mlt.set_start_method('spawn')  # Make multiprocess can use opencv window, for linux

    date = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

    if not os.path.exists(OUTPUT_LOGGING_DIR):
        os.mkdir(OUTPUT_LOGGING_DIR)

    sys.stdout = Logger(os.path.join(OUTPUT_LOGGING_DIR,'{0}.log'.format(date)))

    sys.stderr = Logger_Error(os.path.join(OUTPUT_LOGGING_DIR,'{0}_error.log'.format(date)))

    # assert os.path.exists('/media/sjtu/DISK/'), "DISK NOT LOAD" # 外加载硬盘存在检查，若你未用外挂载硬盘可以删去

    ###### TODO:串口通信设置，你如果暂时不用串口通信，可以将它们注释掉，且该模块只能在linux下运行########
    # password = 'radar'
    # ch = pexpect.spawn('sudo chmod 777 {}'.format(usb))
    # ch.sendline(password)
    # print('set password ok')
    #
    # ser = serial.Serial(usb, 115200, timeout=0.2)
    # if ser.is_open:
    #     print("open ok")
    #     ser.flushInput()
    # else:
    #     ser.open()
    #
    # # 串口读写线程开启
    # _thread.start_new_thread(read, (ser,))
    # _thread.start_new_thread(write, (ser,))
    ##########################################################

    app = QtWidgets.QApplication(sys.argv)
    myshow = Mywindow()
    # 初始化UI界面并展示
    myshow.show()



    # 初始化显示，enemy是否选对
    SELF_COLOR = ['BLUE', 'RED']
    myshow.set_text("message_box", f"You are {SELF_COLOR[enemy]:s}")

    print("=" * 30)
    print("[INFO] Starting.")

    # 这个是用来放在单独显示的opencv窗口进行键盘按键调节和使得UI窗口更新，其内容是什么无所谓
    demo_frame = cv2.imread(INIT_FRAME_PATH)
    cv2.namedWindow("out", cv2.WINDOW_NORMAL)

    # hp_scene init
    hp_scene = HP_scene(enemy, lambda x: myshow.set_image(x, "message_box"))
    # saving log
    if not battle_mode:
        if not os.path.exists(SPECIFIC_LOGGING_NON_BATTLE_MODE_DIR):
            os.mkdir(SPECIFIC_LOGGING_NON_BATTLE_MODE_DIR)

        title = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

        f = open(os.path.join(SPECIFIC_LOGGING_NON_BATTLE_MODE_DIR, f"{title:s}.log"), 'w')
    else:
        if not os.path.exists(SPECIFIC_LOGGING_BATTLE_MODE_DIR):
            os.mkdir(SPECIFIC_LOGGING_BATTLE_MODE_DIR)

        title = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

        f = open(os.path.join(SPECIFIC_LOGGING_BATTLE_MODE_DIR, f"{title:s}_battle.log"), 'w')

    try:
        main_process = radar_process(myshow, f)
        open_base_flag = False
        key_close = False
        while True:
            t1 = time.time()

            UART_passer.get_message(hp_scene) # 显示hp信息

            hp_scene.show()  # update the message box

            # 子程序置位符
            base_open = False
            base_close = False
            close_flag = False

            if UART_passer.change_view:
                myshow.btn2_on_clicked()
                UART_passer.change_view = False # reset

            if UART_passer.One_compete_start():
                myshow.showFullScreen() # 比赛开始时全屏
                main_process.get_position_using_last() # 若未设置位姿，用上一次位姿
                if open_base_flag:
                    myshow.btn4_on_clicked() # 当一场比赛结束时，关闭子视野（对应全屏）
                if not myshow.record_state:
                    myshow.btn1_on_clicked() # 开始录制

            end_flag, remaining_battle = UART_passer.One_compete_end()
            if end_flag:
                myshow.showNormal()  # 一场比赛结束时，恢复小窗
                if not open_base_flag:
                    myshow.btn4_on_clicked() # 当一场比赛结束时，开启子视野（对应非全屏）
                if myshow.record_state:
                    myshow.btn1_on_clicked() # 结束录制

            # only perform when the stage change 即open_base_flag表示当前子进程窗口有没有打开，
            # 然后外部操作使得裁判系统类的open_base与主程序open_base_flag不一样了，就说明要改变，那么在spin_once中改变
            if UART_passer.open_base and not open_base_flag or not UART_passer.open_base and open_base_flag:
                if UART_passer.open_base:
                    myshow.showNormal()
                    base_open = True
                else:
                    myshow.showFullScreen()
                    base_close = True
                open_base_flag = UART_passer.open_base

            if UART_passer.getposition:
                # 位姿估计
                myshow.set_text("feedback", "perform position examine..")
                UART_passer.getposition = False
                main_process.get_position_new()

            # 若比赛局数到最大（这里我们没考虑主裁判终止比赛这种情况，不过这种情况很难判断，而且也不多）或者通过opencv窗口键盘输入q结束程序
            if remaining_battle == 0 or key_close:
                close_flag = True
                if not using_dq:
                    Radar.start_record()  # make radar recording
                myshow.set_text("feedback", "[INFO] game END")

            main_process.spin_once(base_open, base_close, close_flag, is_opening=open_base_flag)

            if close_flag:
                main_process.join() # 子进程终止
                break

            fps = 1 / (time.time() - t1)
            f.write(f"fps:{fps:.03f}\n")
            if not battle_mode:
                print("fps: {0:.3f}".format(fps))

            cv2.imshow("out", demo_frame)

            k = cv2.waitKey(1) # there to change the playing rate
            # 这些都是给雷达站准备人员，云台手操作不了
            if k == 0xff & ord("q"):
                key_close = True

            elif k == 0xff & ord("a"):
                myshow.btn2_on_clicked() # 切换视角

            elif k == 0xff & ord('g'):
                UART_passer.anti_dart = True # 开启反导第二阶段

            elif k == 0xff & ord("p"): # 暂停
                cv2.waitKey(0)
    except Exception as e:
        # 出现异常自动结束，并在stderr中记录异常，可查看
        traceback.print_exc()
        myshow.set_text("feedback", "[ERROR] Program broken")

    # 若在录制过程结束录制
    if myshow.record_state:
        myshow.btn1_on_clicked()

    print("=" * 30)
    print("[INFO] Finished.")

    if not using_dq:
        Radar.stop()

    f.close()

    cv2.destroyWindow("out")

    sys.exit(app.exec_())
