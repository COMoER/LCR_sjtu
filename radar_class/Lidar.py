'''
雷达类
基于官方ROS驱动，本脚本只提供订阅"/livox/lidar"topic的后续处理
'''
# 为了某些情况ros驱动没有正常加载，但也可以用主程序类测试
try:
    import rospy
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs import point_cloud2
except:
    print("[ERROR] ROS environment hasn't been successfully loaded.You can only use DepthQueue with saved PointCloud")
import cv2
import os
import numpy as np
import threading
from queue import Queue
import ctypes
import inspect
import time
from datetime import datetime
import pickle as pkl

from radar_class.config import PC_STORE_DIR,LIDAR_TOPIC_NAME

class DepthQueue(object):
    def __init__(self,capacity,size,K_0,C_0,E_0):
        '''
        用队列关系储存点云

        :param capacity: the maximum length of depth queue
        :param size: image size [W,H]
        :param K_0: 相机内参
        :param C_0: 畸变系数
        :param E_0: 雷达到相机外参
        '''
        self.size = size
        self.depth = np.ones((size[1],size[0]),np.float64)*np.nan
        self.queue = Queue(capacity)
        self.K_0 = K_0
        self.C_0 = C_0
        self.rvec = cv2.Rodrigues(E_0[:3,:3])[0]
        self.tvec = E_0[:3,3]
        self.E_0 = E_0
        self.init_flag = False

    def push_back(self,pc:np.array):

        # 当队列为空时，说明该类正在被初始化，置位初始化置位符
        if self.queue.empty():
            self.init_flag = True

        # pc (N,3) 原始点云

        # 坐标转换 由雷达坐标转化相机坐标，得到点云各点在相机坐标系中的z坐标
        dpt = (self.E_0@(np.concatenate([pc,np.ones((pc.shape[0],1))],axis = 1).transpose())).transpose()[:,2]

        # 得到雷达点云投影到像素平面的位置
        ip = cv2.projectPoints(pc,self.rvec,self.tvec,self.K_0,self.C_0)[0].reshape(-1,2).astype(np.int)

        # 判断投影点是否在图像内部
        inside = np.logical_and(np.logical_and(ip[:, 0] >= 0, ip[:, 0] < self.size[0]),
                                np.logical_and(ip[:, 1] >= 0, ip[:, 1] < self.size[1]))
        ip = ip[inside]
        dpt = dpt[inside]

        # 将各个点的位置[N,2]加入队列
        self.queue.put(ip)
        if self.queue.full():
            # 队满，执行出队操作，将出队的点云中所有点对应的投影位置的值置为nan
            ip_d = self.queue.get()
            self.depth[ip_d[:, 1], ip_d[:, 0]] = np.nan

        #TODO: 如果点云有遮挡关系，则测距测到前或后不确定

        # 更新策略，将进队点云投影点的z值与原来做比较，取较小的那个
        s = np.stack([self.depth[ip[:,1],ip[:,0]],dpt],axis = 1)
        s = np.nanmin(s, axis=1)
        self.depth[ip[:,1],ip[:,0]] = s

    def depth_detect_refine(self,r):
        '''
        :param r: the bounding box of armor , format (x0,y0,w,h)

        :return: (x0,y0,z) x0,y0是中心点在归一化相机平面的坐标前两位，z为其对应在相机坐标系中的z坐标值
        '''
        center = np.float32([r[0]+r[2]/2,r[1]+r[3]/2])
        # 采用以中心点为基准点扩大一倍的装甲板框，并设置ROI上界和下界，防止其超出像素平面范围
        area = self.depth[int(max(0,center[1]-r[3])):int(min(center[1]+r[3],self.size[1]-1)),
               int(max(center[0]-r[2],0)):int(min(center[0]+r[2],self.size[0]-1))]

        z = np.nanmean(area) if not np.isnan(area).all() else np.nan # 当对应ROI全为nan，则直接返回为nan

        return np.concatenate([cv2.undistortPoints(center, self.K_0, self.C_0).reshape(-1),np.array([z])],axis = 0)
    def detect_depth(self,rects):
        '''
        :param rects: List of the armor bounding box with format (x0,y0,w,h)

        :return: an array, the first dimension is the amount of armors input, and the second is the location data (x0,y0,z)
        x0,y0是中心点在归一化相机平面的坐标前两位，z为其对应在相机坐标系中的z坐标值
        '''
        if len(rects) == 0:
            return []

        ops = []

        for rect in rects:
             ops.append(self.depth_detect_refine(rect))

        return np.stack(ops,axis = 0)

# 安全关闭子线程
def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


class Radar(object):

    # the global member of the Radar class
    __init_flag = False # 雷达启动标志
    __working_flag = False # 雷达接收线程启动标志
    __threading = None # 雷达接收子线程

    __lock = threading.Lock() # 线程锁
    __queue = [] # 一个列表，存放雷达类各个对象的Depth Queue

    __record_times = 0 # 已存点云的数量

    __record_list = []

    __record_max_times = 100 # 最大存点云数量

    def __init__(self,K_0,C_0,E_0,queue_size = 200,imgsz = (3088, 2064)):
        '''
        雷达处理类，对每个相机都要创建一个对象

        :param K_0:相机内参
        :param C_0:畸变系数
        :param E_0:雷达到相机外参
        :param queue_size:队列最大长度
        :param imgsz:相机图像大小
        '''
        if not Radar.__init_flag:
            # 当雷达还未有一个对象时，初始化接收节点
            Radar.__laser_listener_begin(LIDAR_TOPIC_NAME)
            Radar.__init_flag = True
            Radar.__threading=threading.Thread(target = Radar.__main_loop,daemon=True)
        self._no = len(Radar.__queue) # 该对象对应于整个雷达对象列表的序号
        self._K_0 = K_0
        self._C_0 = C_0
        Radar.__queue.append(DepthQueue(queue_size,imgsz,K_0,C_0,E_0))

    @staticmethod
    def start():
        '''
        开始子线程，即开始spin
        '''
        if not Radar.__working_flag:
            Radar.__threading.start()
            Radar.__working_flag = True
    @staticmethod
    def stop():
        '''
        结束子线程
        '''
        if Radar.__working_flag:
            stop_thread(Radar.__threading)
            Radar.__working_flag = False

    @staticmethod
    def __callback(data):
        '''
        子线程函数，对于/livox/lidar topic数据的处理
        '''
        if Radar.__working_flag:
            Radar.__lock.acquire()

            pc = np.float32(point_cloud2.read_points_list(data, field_names=("x", "y", "z"),skip_nans=True)).reshape(-1,3)

            dist = np.linalg.norm(pc,axis = 1)

            pc = pc[dist>0.4] # 雷达近距离滤除
            # do record
            if Radar.__record_times > 0:

                Radar.__record_list.append(pc)
                print("[INFO] recording point cloud {0}/{1}".format(Radar.__record_max_times - Radar.__record_times,Radar.__record_max_times))
                if Radar.__record_times == 1:
                    try:
                        if not os.path.exists(PC_STORE_DIR):
                            os.mkdir(PC_STORE_DIR)
                        with open("{0}/{1}.pkl"
                                          .format(PC_STORE_DIR,
                                                  datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M')),
                                  'wb') as f:
                            pkl.dump(Radar.__record_list, f)
                        Radar.__record_list.clear()
                        print("[INFO] record finished")
                    except:  # 当出现磁盘未挂载等情况，导致文件夹都无法创建
                        print("[ERROR] The point cloud save dir even doesn't exist on this computer!")
                Radar.__record_times -= 1
            # update every class object's queue
            for q in Radar.__queue:
                q.push_back(pc)

            Radar.__lock.release()

    @staticmethod
    def __laser_listener_begin(laser_node_name = "/livox/lidar"):
        rospy.init_node('laser_listener', anonymous=True)
        rospy.Subscriber(laser_node_name, PointCloud2,Radar.__callback)
    @staticmethod
    def __main_loop():
        # 通过将spin放入子线程来防止其对主线程的阻塞
        rospy.spin()
        # 当spin调用时，subscriber就会开始轮询接收所订阅的节点数据，即不断调用callback函数

    @staticmethod
    def start_record():
        '''
        开始录制点云
        '''
        if Radar.__record_times == 0:
            Radar.__record_times = Radar.__record_max_times

    def detect_depth(self,rects):
        '''
        接口函数，传入装甲板bounding box返回对应（x0,y0,z_c)值
        ps:这个x0,y0是归一化相机坐标系中值，与下参数中指代bounding box左上方点坐标不同

        :param rects: armor bounding box, format: (x0,y0,w,h)
        '''
        Radar.__lock.acquire()
        # 通过self.no来指定该对象对应的深度队列
        results = Radar.__queue[self._no].detect_depth(rects)
        Radar.__lock.release()
        return results

    def read(self):
        '''
        debug用，返回深度队列当前的深度图
        '''
        Radar.__lock.acquire()
        depth = Radar.__queue[self._no].depth.copy()
        Radar.__lock.release()
        return depth

    def check_radar_init(self):
        '''
        检查该队列绑定队列置位符，来确定雷达是否正常工作
        '''
        if Radar.__queue[self._no].init_flag:
            Radar.__queue[self._no].init_flag = False
            return True
        else:
            return False

    def __del__(self):
        Radar.stop()


if __name__ == '__main__':
    # 测试demo 同时也是非常好的测距测试脚本
    from radar_class.camera import read_yaml,Camera_Thread
    from radar_class import locate_pick
    import traceback

    _,K_0,C_0,E_0,imgsz = read_yaml(0)

    ra = Radar(K_0,C_0,E_0,imgsz=imgsz)
    Radar.start()

    cv2.namedWindow("out",cv2.WINDOW_NORMAL) # 显示雷达深度图
    cv2.namedWindow("img",cv2.WINDOW_NORMAL) # 显示实际图片
    cap = Camera_Thread(0)
    try:
        flag, frame = cap.read()

        # 选定一个ROI区域来测深度
        cv2.imshow("img", frame)
        rect = cv2.selectROI("img", frame, False)


        _,rvec,tvec = locate_pick(cap,0,0) # 用四点手动标定估计位姿

        # 创建transform matrix
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.reshape(-1)
        T = np.linalg.inv(T)

        key = cv2.waitKey(1)

        while (flag and key != ord('q') & 0xFF):

            depth = ra.read() # 获得深度图

            # 分别在实际相机图和深度图上画ROI框来对照
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

            cv2.rectangle(depth, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 255, 3)

            cv2.imshow("out", depth)
            cv2.imshow("img", frame)

            key = cv2.waitKey(1)
            if key == ord('r') & 0xFF:
                # 重选区域
                rect = cv2.selectROI("img", frame, False)

            if key == ord('s') & 0xFF:
                # 显示世界坐标系和相机坐标系坐标和深度，以对测距效果进行粗略测试
                cp = ra.detect_depth([rect]).reshape(-1)

                cp = ( T @ np.concatenate(
                    [np.concatenate([cp[:2], np.ones(1)], axis=0) * cp[2], np.ones(1)], axis=0))[:3]

                cp_eye = (np.eye(4) @ np.concatenate(
                    [np.concatenate([cp[:2], np.ones(1)], axis=0) * cp[2], np.ones(1)], axis=0))[:3]

                print(f"target position is ({cp[0]:0.3f},{cp[1]:0.3f},{cp[2]:0.3f}) and distance is {np.linalg.norm(cp):0.3f}")

                print(f"origin target position is ({cp_eye[0]:0.3f},{cp_eye[1]:0.3f},{cp_eye[2]:0.3f}) and distance is {np.linalg.norm(cp_eye):0.3f}")

            flag, frame = cap.read()
    except:
        traceback.print_exc()
    Radar.stop()
    cap.release()
    cv2.destroyAllWindows()


