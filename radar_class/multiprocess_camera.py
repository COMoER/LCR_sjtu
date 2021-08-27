'''
相机多进程类

值得注意的是，由于ubuntu一些特性，导致如果你主看了UI界面，这时候子进程窗口不会显示在最前，对于雷达这种自动机器人，这是致命的，所以在子进程弹出窗口，
主UI必须小窗化并且主UI在左，子进程窗口在右，主UI窗口初始位置需要雷达站设置人员在比赛准备时留意
'''
import cv2
import os
import time
from datetime import datetime
import multiprocessing as mlt

from radar_class.camera import Camera_Thread,read_yaml
from radar_class.config import THIRD_CAMERA_SAVE_DIR,using_video,THIRD_VIDEO_PATH,win_size

def base_process(recv_queue: mlt.Queue, message):
    '''
    多进程函数

    :param recv_queue:多进程消息队列
    :param message:多进程共享数据字典
    '''
    imgsz = read_yaml(2)[4]

    # 屏幕大小

    if message["battle_mode"]:
        try:
            if not os.path.exists(THIRD_CAMERA_SAVE_DIR):
                os.mkdir(THIRD_CAMERA_SAVE_DIR)
        except:  # 当出现磁盘未挂载等情况，导致文件夹都无法创建
            print("[ERROR] The third video save dir even doesn't exist on this computer!")
        writer = cv2.VideoWriter(os.path.join(THIRD_CAMERA_SAVE_DIR,"{0}.mp4".format(datetime.now().strftime('%Y-%m-%d %H-%M-%S'))),
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, imgsz)

    cap = Camera_Thread(2,using_video,THIRD_VIDEO_PATH)

    cap_init_open = False
    # Due to we will stop this process when no task, so we should ensure it open the camera at least once

    if cap.is_open():
        print("[INFO] Camera {0} Starting for viewing base.".format(2))
        cap_init_open = True
    else:
        print("[INFO] Camera {0} Failed, try to open.".format(2))

    # start get image
    # 以下过程也可见流程图
    task = 0
    while True:
        if task == 0:
            # only when former task finished perform the next task
            task = recv_queue.get() # waiting the message pushed
            if task == 1:
                # create opencv window
                cv2.namedWindow("base shoot", cv2.WINDOW_NORMAL)
                w = 530
                h = 480
                cv2.resizeWindow("base shoot", w, h)
                width,height = win_size
                cv2.moveWindow("base shoot", width - w, height // 2 - h // 2)
                cv2.setWindowProperty("base shoot", cv2.WND_PROP_TOPMOST, 1)
                # in the next loop, the task is 1
            if task == 2:  # 2 is close
                break
        if task == 1:
            if not recv_queue.empty(): # If there is a message in the queue, we must check it at once.If not, go on to get the image
                task = recv_queue.get()
                if task == 2:  # 2 is close
                    break
                if task == 0:
                    cv2.destroyWindow("base shoot")
                continue
            # If camera failed, try to reopen
            if not cap.is_open():
                cap.open()
                if not cap_init_open and cap.is_open():
                    cap_init_open = True
            # when task 1 , start showing
            if not cap_init_open: # the camera hasn't initialized yet
                message['flag'] = False
                time.sleep(0.05)
                continue
            r, frame = cap.read()
            if not r:
                message['flag'] = False
                time.sleep(0.05)
                continue
            if message["battle_mode"] and message["record"]:
                writer.write(frame)
            cv2.imshow("base shoot", frame)
            cv2.waitKey(1)
            message['flag'] = True
    cap.release()
    writer.release()
    print("process base finished")