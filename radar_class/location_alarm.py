'''
位置预警类
'''
import cv2
import numpy as np

from radar_class.config import armor_list, color2enemy, enemy_case, MAP_PATH,map_size
from radar_class.common import is_inside

# 特别提示，本脚本中对于装甲板预测出的装甲板编号(cls)解析，均采用R1-R5编号为8-12,B1-B5为1-5来解析，请用自己装甲板预测器进行预测时，进行编号转换

class CompeteMap(object):
    '''
    小地图绘制类

    使用顺序 twinkle->update->show->refresh
    '''
    # arguments
    _circle_size = 10  # 圈大小
    _twinkle_times = 3  # 闪烁几次

    def __init__(self, region, real_size, enemy, api):
        """
        :param region:预警区域
        :param real_size:真实赛场大小
        :param enemy:敌方编号
        :param api:显示api f(img)
        """
        self._enemy = enemy
        # map为原始地图(canvas),out_map在每次refresh时复制一份canvas副本并在其上绘制车辆位置及预警
        self._map = cv2.imread(MAP_PATH)
        self._map = cv2.resize(self._map, map_size)
        # 显示api调用（不要跨线程）
        self._show_api = api
        self._real_size = real_size  # 赛场实际大小
        self._draw_region(region)
        # 闪烁画面，out_map前一步骤，因为out_map翻转过，而region里面（x,y）依照未翻转的坐标定义，若要根据region进行闪烁绘制，用未翻转地图更加方便
        self._out_map_twinkle = self._map.copy()
        # 画点以后画面
        if self._enemy:
            # enemy is blue,逆时针旋转90
            self._out_map = cv2.rotate(self._out_map_twinkle, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # enemy is blue,顺时针旋转90
            self._out_map = cv2.rotate(self._out_map_twinkle, cv2.ROTATE_90_CLOCKWISE)

        self._twinkle_event = {}

    def _refresh(self):
        """
        闪烁画面复制canvas,刷新
        """
        self._out_map_twinkle = self._map.copy()

    def _update(self, location: dict):
        '''
        更新车辆位置

        :param location:车辆位置字典 索引为'1'-'10',内容为车辆位置数组(2,)
        '''
        if self._enemy:
            self._out_map = cv2.rotate(self._out_map_twinkle, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            self._out_map = cv2.rotate(self._out_map_twinkle, cv2.ROTATE_90_CLOCKWISE)
        _loc_map = [0, 0]
        for armor in location.keys():
            ori_x = int(location[armor][0] / self._real_size[0] * map_size[0])
            ori_y = int((self._real_size[1] - location[armor][1]) / self._real_size[1] * map_size[1])
            # 位置翻转
            if self._enemy:
                _loc_map = ori_y, map_size[0] - ori_x
            else:
                _loc_map = map_size[1] - ori_y, ori_x
            # 画定位点
            self._draw_circle(_loc_map, int(armor))

    def _show(self):
        '''
        调用show_api展示
        '''
        self._show_api(self._out_map)

    def _draw_region(self, region: dict):
        '''
        在canvas绘制预警区域
        '''
        for r in region.keys():
            ftype, alarm_type, _, _, _ = r.split('_')
            if ftype == 'm' or ftype == 'a':  # 预警类型判断，若为map或all类型
                if alarm_type == 'l':
                    # 直线预警
                    rect = region[r]  # 获得直线两端点，为命名统一命名为rect
                    # 将实际世界坐标系坐标转换为地图上的像素位置
                    cv2.line(self._map, (int(rect[0] * map_size[0] // self._real_size[0]),
                                         int((self._real_size[1] - rect[1]) * map_size[1] // self._real_size[1])),
                             (int(rect[2] * map_size[0] // self._real_size[0]),
                              int((self._real_size[1] - rect[3]) * map_size[1] // self._real_size[1])),
                             (0, 255, 0), 2)
                if alarm_type == 'r':
                    # 矩形预警
                    rect = region[r]  # rect(x0,y0,x1,y1)
                    cv2.rectangle(self._map,
                                  (int(rect[0] * map_size[0] // self._real_size[0]),
                                   int((self._real_size[1] - rect[1]) * map_size[1] // self._real_size[1])),
                                  (int(rect[2] * map_size[0] // self._real_size[0]),
                                   int((self._real_size[1] - rect[3]) * map_size[1] // self._real_size[1])),
                                  (0, 255, 0), 2)
                if alarm_type == 'fp':
                    # 四点预警
                    f = lambda x: (int(x[0] * map_size[0] // self._real_size[0]),
                                   int((self._real_size[1] - x[1]) * map_size[1] // self._real_size[1]))
                    rect = np.array(region[r][:8]).reshape(4, 2)  # rect(x0,y0,x1,y1)
                    for i in range(4):
                        cv2.line(self._map, f(rect[i]), f(rect[(i + 1) % 4]), (0, 255, 0), 2)

    def _draw_circle(self, location, armor: int):
        '''
        画定位点
        '''
        img = self._out_map
        color = (255 * (armor // 6), 0, 255 * (1 - armor // 6))  # 解算颜色
        armor = armor - 5 * (armor > 5)  # 将编号统一到1-5
        cv2.circle(img, tuple(location), self._circle_size, color, -1)  # 内部填充
        cv2.circle(img, tuple(location), self._circle_size, (0, 0, 0), 1)  # 外边框
        # 数字
        cv2.putText(img, str(armor),
                    (location[0] - 7 * self._circle_size // 10, location[1] + 6 * self._circle_size // 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self._circle_size / 10, (255, 255, 255), 2)

    def _add_twinkle(self, region: str):
        '''
        预警添加函数

        :param region:要预警的区域名
        '''
        if region not in self._twinkle_event.keys():
            # 若不在预警字典内，则该事件从未被预警过，添加预警事件
            # 预警字典内存有各个预警项目的剩余预警次数(*2,表示亮和灭)
            self._twinkle_event[region] = self._twinkle_times * 2
        else:
            if self._twinkle_event[region] % 2 == 0:
                # 剩余预警次数为偶数，当前灭，则添加至最大预警次数
                self._twinkle_event[region] = self._twinkle_times * 2
            else:
                # 剩余预警次数为奇数，当前亮，则添加至最大预警次数加一次灭过程
                self._twinkle_event[region] = self._twinkle_times * 2 + 1

    def _twinkle(self, region):
        '''
        闪烁执行类

        :param region:所有预警区域
        '''
        for r in self._twinkle_event.keys():
            if self._twinkle_event[r] == 0:
                # 不能再预警
                continue
            if self._twinkle_event[r] % 2 == 0:
                # 当前灭，且还有预警次数，使其亮
                _, alarm_type, _, _, _ = r.split('_')
                # 闪
                if alarm_type == 'r':
                    rect = region[r]  # rect(x0,y0,x1,y1)
                    cv2.rectangle(self._out_map_twinkle,
                                  (int(rect[0] * map_size[0] // self._real_size[0]),
                                   int((self._real_size[1] - rect[1]) * map_size[1] // self._real_size[1])),
                                  (int(rect[2] * map_size[0] // self._real_size[0]),
                                   int((self._real_size[1] - rect[3]) * map_size[1] // self._real_size[1])),
                                  (0, 0, 255), -1)
                if alarm_type == 'fp':
                    x = np.float32(region[r][:8]).reshape(4, 2)
                    x[:, 0] = (x[:, 0] * map_size[0] // self._real_size[0])
                    x[:, 1] = ((self._real_size[1] - x[:, 1]) * map_size[1] // self._real_size[1])

                    rect = x.astype(int)
                    cv2.fillConvexPoly(self._out_map_twinkle, rect, (0, 0, 255))
            # 减少预警次数
            self._twinkle_event[r] -= 1


class Alarm(CompeteMap):
    '''
    预警类，继承自地图画图类
    '''
    # param

    _pred_time = 10  # 预测几次
    _pred_radio = 0.2  # 预测速度比例

    _ids = {1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5}  # 装甲板编号到标准编号
    _lp = True  # 是否位置预测
    _z_a = True  # 是否进行z轴突变调整
    _z_thre = 0.2  # z轴突变调整阈值
    _ground_thre = 100  # 地面阈值，我们最后调到了100就是没用这个阈值，看情况调
    _using_l1 = True  # 不用均值，若两个都有预测只用右相机预测值

    def __init__(self, region: dict, api, touch_api, enemy, real_size, two_camera=True, debug=False):
        '''

        :param region:预警区域
        :param api:主程序显示api，传入画图程序进行调用（不跨线程使用,特别是Qt）
        :param touch_api:车间通信api
        :param enemy:敌方编号
        :param real_size:场地实际大小
        :param two_camera:是否使用两个相机
        :param debug:debug模式
        '''
        super(Alarm, self).__init__(region, real_size, enemy, api)
        self._region = region
        self._location = {}
        if two_camera:
            # 分别为z坐标缓存，相机世界坐标系位置，以及（相机到世界）转移矩阵
            self._z_cache = [None, None]
            self._camera_position = [None, None]
            self._T = [None, None]
        else:
            self._z_cache = [None]
            self._camera_position = [None]
            self._T = [None]

        self._location_pred_time = np.zeros(10, dtype=int)  # 预测次数记录
        self._enemy = enemy
        self._touch_api = touch_api
        self._debug = debug
        self._two_camera = two_camera

        # 判断x各行是否为全零的函数
        self._f_equal_zero = lambda x: np.isclose(np.sum(x, axis=1), np.zeros(x.shape[0]))
        for i in range(1, 11):  # 初始化位置为全零
            self._location[str(i)] = [0, 0]
        # 前两帧位置为全零
        self._location_cache = [self._location.copy(), self._location.copy()]

    def push_T(self, T, camera_position, camera_type):
        '''
        位姿信息

        :param T:相机到世界转移矩阵
        :param camera_position:相机在世界坐标系坐标
        :param camera_type:相机编号，若为单相机填0
        '''
        if camera_type > 0 and not self._two_camera:
            return

        self._camera_position[camera_type] = camera_position.copy()
        self._T[camera_type] = T.copy()

    def _check_alarm(self):
        '''
        预警检测

        :return:alarming:各区域是否有预警;
        base_alarming:基地是否有预警
        '''
        alarming = False  #
        base_alarming = False
        for loc in self._region.keys():
            ftype, alarm_type, team, target, l_type = loc.split('_')
            targets = []
            for armor in list(self._location.keys())[0 + self._enemy * 5:5 + self._enemy * 5]:  # 检测敌方
                l = np.float32(self._location[armor])
                if ftype == 'm' or ftype == 'a':  # 若为位置预警
                    if alarm_type == 'r' and (target not in enemy_case or color2enemy[team] == self._enemy):  # 与反投影相同
                        # 矩形区域采用范围判断
                        if l[0] >= self._region[loc][0] and l[1] >= self._region[loc][3] and \
                                l[0] <= self._region[loc][2] and l[1] <= self._region[loc][1]:
                            targets.append(int(armor) - 1)  # targets为0-9
                    if alarm_type == 'l' and color2enemy[team] != self._enemy:  # base alarm
                        # 直线检测，目前我们只用来检测基地，实际上原本意图是用来检测任意斜向区域，但被四点预警区域替代了
                        up_p = np.float32(self._region[loc][:2])  # 上端点
                        dw_p = np.float32(self._region[loc][2:4])  # 下端点
                        dis_thres = self._region[loc][4]
                        up_l = up_p - dw_p  # 直线向上的向量
                        dw_l = dw_p - up_p  # 直线向下的向量
                        m_r = np.float32([up_l[1], -up_l[0]])  # 方向向量，向右
                        m_l = np.float32([-up_l[1], up_l[0]])  # 方向向量，向左
                        f_dis = lambda m: m @ (l - dw_p) / np.linalg.norm(m)  # 计算从下端点到物体点在各方向向量上的投影
                        if l_type == 'l':
                            dis = f_dis(m_l)
                        if l_type == 'r':
                            dis = f_dis(m_r)
                        if l_type == 'a':
                            dis = abs(f_dis(m_r))  # 绝对距离
                        # 当物体位置在线段内侧，且距离小于阈值时，预警
                        if up_l @ (l - dw_p) > 0 and dw_l @ (l - up_p) > 0 and \
                                dis_thres >= dis >= 0:
                            targets.append(int(armor) - 1)
                    if alarm_type == 'fp' and (target not in enemy_case or color2enemy[team] == self._enemy):
                        # 判断是否在凸四边形内
                        if is_inside(np.float32(self._region[loc][:8]).reshape(4, 2), point=l):
                            targets.append(int(armor) - 1)

            if len(targets):
                # 发送预警，跟反投影预警一样
                if alarm_type == 'l':
                    # 基地预警发送，编码规则详见主程序类send_judge
                    base_alarming = True
                    self._touch_api({'task': 3, 'data': [targets]})
                else:
                    super(Alarm, self)._add_twinkle(loc)
                    alarming = True
                    if target in ['feipo', 'feipopre', 'gaodipre', 'gaodipre2']:
                        self._touch_api({'task': 2, 'data': [team, target, targets]})

        return alarming, base_alarming

    def refresh(self):
        '''
        刷新地图
        '''
        super(Alarm, self)._refresh()

    def show(self):
        '''
        执行预警闪烁并画点显示地图
        '''
        super(Alarm, self)._twinkle(self._region)
        super(Alarm, self)._update(self._location)
        super(Alarm, self)._show()

    def _adjust_z_one_armor(self, l, camera_type):
        '''
        z轴突变调整，仅针对一个装甲板

        :param l:(cls+x+y+z) 一个id的位置
        :param camera_type:相机编号
        '''
        if isinstance(self._z_cache[camera_type], np.ndarray):
            mask = np.array(self._z_cache[camera_type][:, 0] == l[0])  # 检查上一帧缓存z坐标中有没有对应id
            if mask.any():
                z_0 = self._z_cache[camera_type][mask][:, 1]
                if z_0 < self._ground_thre:  # only former is on ground do adjust
                    z = l[3]
                    if z - z_0 > self._z_thre:  # only adjust the step from down to up
                        # 以下计算过程详见技术报告公式
                        ori = l[1:].copy()
                        line = l[1:] - self._camera_position[camera_type]
                        radio = (z_0 - self._camera_position[camera_type][2]) / line[2]
                        new_line = radio * line
                        l[1:] = new_line + self._camera_position[camera_type]
                        if self._debug:
                            # z轴变换debug输出
                            print('{0} from'.format(armor_list[(self._ids[int(l[0])]) - 1]), ori, 'to', l[1:])

    def _location_prediction(self):
        '''
        位置预测
        '''

        # 次数统计
        time_equal_one = self._location_pred_time == 1  # 若次数为1则不能预测
        time_equal_zero = self._location_pred_time == 0

        # 上两帧位置 (2,N)
        pre = np.stack([np.float32(list(self._location_cache[0].values())),
                        np.float32(list(self._location_cache[1].values()))], axis=0)
        # 该帧预测位置
        now = np.float32(list(self._location.values()))

        pre2_zero = self._f_equal_zero(pre[0])  # the third latest frame 倒数第二帧
        pre1_zero = self._f_equal_zero(pre[1])  # the second latest frame 倒数第一帧
        now_zero = self._f_equal_zero(now)  # the latest frame 当前帧

        # 仅对该帧全零，上两帧均不为0的id做预测
        do_prediction = np.logical_and(
            np.logical_and(np.logical_and(np.logical_not(pre2_zero), np.logical_not(pre1_zero)),
                           now_zero), np.logical_not(time_equal_one))
        v = self._pred_radio * (pre[1] - pre[0])  # move vector between frame
        if self._debug:
            # 被预测id,debug输出
            for i in range(10):
                if do_prediction[i]:
                    print("{0} lp yes".format(armor_list[i]))
        now[do_prediction] = v[do_prediction] + pre[1][do_prediction]

        set_time = np.logical_and(do_prediction, time_equal_zero)  # 次数为0且该帧做预测的设置为最大次数
        reset = np.logical_and(np.logical_not(now_zero), time_equal_one)  # 对当前帧不为0，且次数为1的进行次数重置
        self._location_pred_time[reset] = 0
        self._location_pred_time[set_time] = self._pred_time + 1
        self._location_pred_time[do_prediction] -= 1  # 对做预测的进行次数衰减

        # 预测填入
        for i in range(1, 11):
            self._location[str(i)] = now[i - 1].tolist()

        # push new data
        self._location_cache[0] = self._location_cache[1].copy()
        self._location_cache[1] = self._location.copy()

    def check(self):
        '''
        预警检测
        '''
        alarming, base_alarming = self._check_alarm()

        return alarming, base_alarming

    def two_camera_merge_update(self, locations, extra_locations, radar):
        """
        两个相机合并更新，顾名思义，two_camera为True才能用的专属api

        :param locations: the list of the predicted locations [N,fp+conf+cls+img_no+bbox] of the both two cameras
        :param extra_locations:the list of real_scene class output using bbox prediction [:,cls+bbox] of the both two cameras
        :param radar:  the list of the radar class corresponding to the two camera
        """
        if self._two_camera:
            # init location
            for i in range(1, 11):
                self._location[str(i)] = [0, 0]
            rls = []
            ex_rls = []
            for location, e_location, ra in zip(locations, extra_locations, radar):
                # 针对每一个相机产生的结果

                # 对于用神经网络直接预测出的装甲板，若不为None
                if isinstance(location, np.ndarray):
                    l = ra.detect_depth(location[:, 11:])  # (N,x0+y0+z)  z maybe nan
                    # nan滤除
                    mask = np.logical_not(np.any(np.isnan(l), axis=1))
                    # 格式为 (N,cls+x0+y0+z)
                    rls.append(np.concatenate([location[mask].reshape(-1, 15)[:, 9].reshape(-1, 1),
                                               l[mask].reshape(-1, 3)], axis=1))
                else:
                    rls.append(None)

                # 同上，不过这里是对IoU预测的装甲板做解析
                if isinstance(e_location, np.ndarray):
                    e_l = ra.detect_depth(e_location[:, 1:])
                    # nan滤除
                    mask = np.logical_not(np.any(np.isnan(e_l), axis=1))
                    ex_rls.append(np.concatenate([e_location[mask].reshape(-1, 5)[:, 0].reshape(-1, 1),
                                                  e_l[mask].reshape(-1, 3)], axis=1))
                else:
                    ex_rls.append(None)
            # 以下判断逻辑按照“如果有直接神经网络预测的装甲板，就直接用它，而不用基于IoU预测出的装甲板”的原则
            pred_loc = []  # 存储预测出的位置 cls+x+y+z
            if self._z_a:  # 两个相机z缓存，存储列表
                pred_1 = []
                pred_2 = []
            for armor in self._ids.keys():
                l1 = None  # 对于特定id，第一个相机基于直接神经网络预测装甲板计算出的位置
                l2 = None  # 对于特定id，第二个相机基于直接神经网络预测装甲板计算出的位置
                el1 = None  # 对于特定id，第一个相机基于IoU预测装甲板计算出的位置
                el2 = None  # 对于特定id，第二个相机基于IoU预测装甲板计算出的位置
                al1 = None  # 对于特定id，第一个相机预测出的位置（直接神经网络与IoU最多有一个预测，不可能两个同时）
                al2 = None  # 对于特定id，第二个相机预测出的位置（直接神经网络与IoU最多有一个预测，不可能两个同时）
                if isinstance(rls[0], np.ndarray):
                    mask = rls[0][:, 0] == armor
                    if mask.any():  # 若有
                        l1 = rls[0][mask].reshape(-1)
                        # 坐标换算为世界坐标
                        l1[1:] = (self._T[0] @ np.concatenate(
                            [np.concatenate([l1[1:3], np.ones(1)], axis=0) * l1[3], np.ones(1)], axis=0))[:3]
                        # z坐标解算
                        if self._z_a:
                            self._adjust_z_one_armor(l1, 0)
                        al1 = l1
                if isinstance(rls[1], np.ndarray):
                    mask = rls[1][:, 0] == armor
                    if mask.any():
                        l2 = rls[1][mask].reshape(-1)
                        l2[1:] = (self._T[1] @ np.concatenate(
                            [np.concatenate([l2[1:3], np.ones(1)], axis=0) * l2[3], np.ones(1)], axis=0))[:3]
                        if self._z_a:
                            self._adjust_z_one_armor(l2, 1)
                        al2 = l2
                if isinstance(ex_rls[0], np.ndarray):
                    mask = ex_rls[0][:, 0] == armor
                    if mask.any():
                        el1 = ex_rls[0][mask].reshape(-1)
                        el1[1:] = (self._T[0] @ np.concatenate(
                            [np.concatenate([el1[1:3], np.ones(1)], axis=0) * el1[3], np.ones(1)], axis=0))[:3]
                        if self._z_a:
                            self._adjust_z_one_armor(el1, 0)
                        al1 = el1
                if isinstance(ex_rls[1], np.ndarray):
                    mask = ex_rls[1][:, 0] == armor
                    if mask.any():
                        el2 = ex_rls[1][mask].reshape(-1)
                        el2[1:] = (self._T[1] @ np.concatenate(
                            [np.concatenate([el2[1:3], np.ones(1)], axis=0) * el2[3], np.ones(1)], axis=0))[:3]
                        if self._z_a:
                            self._adjust_z_one_armor(el2, 1)
                        al2 = el2
                # z cache
                if self._z_a:
                    if isinstance(al1, np.ndarray):
                        pred_1.append(al1[[0, 3]])  # cache cls+z
                    if isinstance(al2, np.ndarray):
                        pred_2.append(al2[[0, 3]])
                # perform merging
                # 参考技术报告，有一些不同，代码里是先进行了z轴调整，不过差不多
                armor_pred_loc = None
                if isinstance(l1, np.ndarray):
                    armor_pred_loc = l1.reshape(-1)
                if isinstance(l2, np.ndarray):
                    if isinstance(armor_pred_loc, np.ndarray):
                        if not self._using_l1:
                            armor_pred_loc = (armor_pred_loc + l2.reshape(-1)) / 2  # 若不用l1，取平均值
                    else:
                        armor_pred_loc = l2.reshape(-1)
                # if not appear in either l1 or l2, then check extra
                if not isinstance(armor_pred_loc, np.ndarray):
                    if isinstance(el1, np.ndarray):
                        armor_pred_loc = el1.reshape(-1)
                    if isinstance(el2, np.ndarray):
                        if isinstance(armor_pred_loc, np.ndarray):
                            if not self._using_l1:
                                armor_pred_loc = (armor_pred_loc + el2.reshape(-1)) / 2
                        else:
                            armor_pred_loc = el2.reshape(-1)
                if isinstance(armor_pred_loc, np.ndarray):
                    pred_loc.append(armor_pred_loc)
            # z cache
            if self._z_a:
                if len(pred_1):
                    self._z_cache[0] = np.stack(pred_1, axis=0)
                else:
                    self._z_cache[0] = None
                if len(pred_2):
                    self._z_cache[1] = np.stack(pred_2, axis=0)
                else:
                    self._z_cache[1] = None
            # 发送裁判系统小地图
            judge_loc = {}
            if len(pred_loc):
                pred_loc = np.stack(pred_loc, axis=0)
                pred_loc[:, 2] = self._real_size[1] + pred_loc[:, 2]  # 坐标变换 向下平移赛场宽度
                for i, armor in enumerate(pred_loc[:, 0]):
                    self._location[str(self._ids[int(armor)])] = pred_loc[i, 1:3].tolist()  # 类成员只存(x,y)信息
                    judge_loc[str(self._ids[int(armor)])] = pred_loc[i, 1:].tolist()  # 发送包存三维信息
            location = {}
            # 执行位置预测
            if self._lp:
                self._location_prediction()
            if self._debug:
                # 位置debug输出
                for armor, loc in judge_loc.items():
                    print("{0} in ({1:.3f},{2:.3f},{3:.3f})".format(armor_list[int(armor) - 1], *loc))
            for i in range(1, 11):
                location[str(i)] = self._location[str(i)].copy()

            # 执行裁判系统发送
            # judge_loc为未预测的位置，作为logging保存，location为预测过的位置，作为小地图发送
            self._touch_api({'task': 1, 'data': [judge_loc, location]})
        else:
            print('[ERROR] This update function only supports two_camera case, using update instead.')

    def update(self, t_location, e_location, radar):
        '''
        与 two_camera_merge_update类似，注释会简略

        :param t_location: the predicted locations [N,fp+conf+cls+img_no+bbox]
        :param e_location: real_scene class output using bbox prediction [:,cls+bbox]
        :param radar:the radar class

        '''

        if not self._two_camera:

            # 位置信息初始化，上次信息已保存至cache
            for i in range(1, 11):
                self._location[str(i)] = [0, 0]

            locations = None
            if isinstance(t_location, np.ndarray):
                l = radar.detect_depth(t_location[:, 11:])  # z maybe nan
                # nan滤除
                mask = np.logical_not(np.any(np.isnan(l), axis=1))
                locations = np.concatenate([t_location[mask].reshape(-1, 15)[:, 9].reshape(-1, 1),
                                            l[mask].reshape(-1, 3)], axis=1)  # (N,cls+x0+y0+z)

            if isinstance(e_location, np.ndarray):
                e_l = radar.detect_depth(e_location[:, 1:])
                # nan滤除
                mask = np.logical_not(np.any(np.isnan(e_l), axis=1))
                ex_rls = np.concatenate([e_location[mask].reshape(-1, 5)[:, 0].reshape(-1, 1),
                                         e_l[mask].reshape(-1, 3)], axis=1)
                if isinstance(locations, np.ndarray):
                    locations = np.concatenate([locations, ex_rls], axis=0)
                else:
                    locations = ex_rls

            judge_loc = {}
            if isinstance(locations, np.ndarray):
                pred_loc = []
                if self._z_a:
                    cache_pred = []
                for armor in self._ids.keys():
                    if (locations[:, 0] == armor).any():
                        l1 = locations[locations[:, 0] == armor].reshape(-1)

                        l1[1:] = (self._T[0] @ np.concatenate(
                            [np.concatenate([l1[1:3], np.ones(1)], axis=0) * l1[3], np.ones(1)], axis=0))[:3]

                        if self._z_a:
                            self._adjust_z_one_armor(l1, 0)
                            cache_pred.append(l1[[0, 3]])
                        pred_loc.append(l1.reshape(-1))
                if len(pred_loc):
                    l = np.stack(pred_loc, axis=0)
                    cls = l[:, 0].reshape(-1, 1)
                    # z cache
                    if self._z_a:
                        self._z_cache[0] = np.stack(cache_pred, axis=0)

                    l[:, 2] = self._real_size[1] + l[:, 2]  # 坐标变换 向下平移赛场宽度

                    for i, armor in enumerate(cls):
                        self._location[str(self._ids[int(armor)])] = l[i, 1:3].tolist()
                        judge_loc[str(self._ids[int(armor)])] = l[i, 1:].tolist()

            if self._lp:
                self._location_prediction()

            # 执行裁判系统发送
            location = {}

            if self._debug:
                # 位置debug输出
                for armor, loc in judge_loc.items():
                    print("{0} in ({1:.3f},{2:.3f},{3:.3f})".format(armor_list[int(armor) - 1], *loc))

            for i in range(1, 11):
                location[str(i)] = self._location[str(i)].copy()
            self._touch_api({'task': 1, 'data': [judge_loc, location]})

        else:
            print(
                '[ERROR] This update function only supports single_camera case, using two_camera_merge_update instead.')
