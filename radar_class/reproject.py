'''
反投影预警类
'''
import cv2
import numpy as np

from radar_class.camera import read_yaml
from radar_class.missile_detect import Missile
from radar_class.common import plot, armor_plot, is_inside
from radar_class.config import color2enemy, enemy_case


class Real_Scene(object):
    '''
    反投影预警
    '''
    # 类全局变量
    _using_bbox = True  # 若为真，则使用装甲板四点的bounding box四点作为落入预警区域的判断依据
    _twinkle_times = 3  # 闪烁几次
    _iou_cache = False  # 存不存用IOU预测出来的装甲板（详见技术报告）
    _iou_thre = 0.8  # 只有高于IoU阈值的才会被预测
    # 装甲板网络预测的编号和实际预测的编号的对应字典，其中包括用颜色预测出的，对于他们 0 is predicted as blue -1 is predicted as red
    _ids = {1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 0: 11, -1: 12}

    def __init__(self, frame, camera_type, region, enemy, real_size, K_0, C_0, touch_api, debug=False):
        '''
        :param frame:初始输入图像
        :param camera_type:相机编号
        :param region:预警区域
        :param enemy:敌方编号
        :param real_size:赛场大小(在家里测试时，可调整）
        :param K_0:相机内参
        :param C_0:相机外参
        :param touch_api:和裁判系统通信api
        :param debug:debug模式，包括上一帧cache框，基于先验预测的装甲板位置，预警提示，以及飞镖预警debug模式开始
        '''
        self._scene = frame.copy()  # 初始化图像
        self._every_scene = frame.copy()  # 实际各帧显示图像
        self._region = region  # 输入的区域
        self._color_bbox = None  # 只预测了颜色的车辆预测框
        frame_shape = read_yaml(camera_type)[4]
        self._size = frame_shape
        # 红色左下方提示区域（针对输入图像相对于3088*2064作调整），提示云台手看小地图预警
        x1, y1 = 100 * frame_shape[0] // 3088, frame_shape[1] - 200 * frame_shape[1] // 2064
        x2, y2 = 200 * frame_shape[0] // 3088, frame_shape[1] - 100 * frame_shape[1] // 2064
        # 预警区域字典，先放入了红色左下方提示区域
        self._scene_region = {'map': np.array([x1, y1, x2, y1, x2, y2, x1, y2]).reshape(4, 2)}
        x1, y1 = 300 * frame_shape[0] // 3088, frame_shape[1] - 200 * frame_shape[1] // 2064
        x2, y2 = 400 * frame_shape[0] // 3088, frame_shape[1] - 100 * frame_shape[1] // 2064
        # 黄色左下方提示区域，飞镖第一阶段预警提示
        self._scene_region['launch'] = np.array([x1, y1, x2, y1, x2, y2, x1, y2]).reshape(4, 2)
        self._K_0 = K_0
        self._C_0 = C_0
        self._enemy = enemy  # 0 is red, 1 is blue
        self._real_size = real_size
        self._cache = None  # 记录上一帧装甲板检测到的对应的父框
        self._debug = debug
        self._twinkle_event = {}  # 闪烁事件字典，记录闪烁剩余次数
        self._touch_api = touch_api
        self._scene_init = False

        # missile alarming
        self._camera_type = camera_type
        if camera_type == 0:  # 只有右相机才做飞镖预警 #TODO:云台手需要主看右相机视角来看黄色预警提示
            self._missile = Missile(self._touch_api, self._enemy, debug)
            self._missile_two_stage = False

    def show_region(self):
        '''
        主要debug用，通过初始化输入图像来看预警区域反投影，在push_T后使用
        '''
        cv2.namedWindow("region", cv2.WINDOW_NORMAL)
        cv2.imshow("region", self._scene)
        cv2.waitKey(0)

    def push_T(self, rvec, tvec):
        '''
        输入相机位姿(rvec,tvec均为PnP输出，即世界到相机）

        :param rvec:旋转向量
        :param tvec:平移向量

        :return: 相机到世界变换矩阵（4*4）, 相机世界坐标
        '''
        self._rvec = rvec
        self._tvec = tvec
        # 基于位姿做反投影，初始化scene_region预警区域字典
        inside = self._plot_region(None)
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.reshape(-1)
        T = np.linalg.inv(T)

        self._inside = inside
        return T, (T @ (np.array([0, 0, 0, 1])))[:3]

    # 双相机反投影区域有交叠，右相机能预警的直接优先右相机预警
    def get_inside(self):
        '''
        反馈给主程序做剔除重叠区域
        '''
        return self._inside, list(self._scene_region.keys())[2:]

    def remove(self, targets):
        '''
        进行重叠区域剔除

        :param targets:需要剔除的区域
        '''
        if self._scene_init:
            for t in targets:
                self._scene_region.pop(t)

    def open_missile_two_stage(self):
        '''
        launch missile two stage
        '''
        if not self._missile_two_stage:
            self._missile_two_stage = True
            self._missile.init_two_stage()

    def close_missile_two_stage(self):
        '''
        close missile two stage

        #TODO:目前貌似没用到，可以添加给云台手手动关第二阶段
        '''
        if self._missile_two_stage:
            self._missile_two_stage = False

    def update(self, frame, results, armors):
        '''
        更新一帧

        :param frame:更新帧原始image
        :param results:车辆预测框，a list of the prediction to each image 各元素格式为(predicted_class,conf_score,bounding box(format:x0,y0,x1,y1))
        :param armors:过滤后的装甲板框 np.ndarray (N,fp+conf+cls+img_no+bbox)
        '''
        self._every_scene = frame

        if self._camera_type == 0:
            if self._scene_init:
                # 飞镖预警
                if self._missile_two_stage:
                    continue_flag, launch_flag = self._missile.detect_two_stage(frame, self._scene_region)
                    # 当时间超过时，就结束二阶段
                    if not continue_flag:
                        self._missile_two_stage = False
                elif self._missile.detect(frame, self._scene_region, 0):
                    self._add_twinkle('launch')
        # 画预测框同时计算仅有颜色的bbox
        self._color_bbox = plot(results, self._every_scene)  # color bbox (x1,y1,x2,y2)
        # 画装甲板
        armor_plot(armors, self._every_scene)
        if self._scene_init:
            self._plot_region(self._every_scene)

    def show(self, show_api, launch_pre=False):
        '''
        基于给定的show_api(类似于f(img))进行展示，不要跨线程！不要跨线程！特别是Qt
        '''
        # 飞镖预警，当右相机报警时，在左相机里面也闪烁
        if self._camera_type == 0 and 'launch' in self._twinkle_event.keys() and (
                self._twinkle_event['launch'] == self._twinkle_times * 2 or
                self._twinkle_event['launch'] == self._twinkle_times * 2 + 1):
            missile_launch_pre = True
        else:
            missile_launch_pre = False
        # 左相机接收右相机闪烁命令，添加闪烁
        if self._camera_type == 1 and launch_pre:
            self._add_twinkle('launch')

        self._twinkle()  # 绘制闪烁
        show_api(self._every_scene)

        return missile_launch_pre

    def show_no_seen(self, launch_pre=False):
        '''
        切换视角时，另一个相机看不到，调用no_seen方法
        '''
        if self._camera_type == 0 and 'launch' in self._twinkle_event.keys() and (
                self._twinkle_event['launch'] == self._twinkle_times * 2 or
                self._twinkle_event['launch'] == self._twinkle_times * 2 + 1):
            missile_launch_pre = True
        else:
            missile_launch_pre = False

        if self._camera_type == 1 and launch_pre:
            self._add_twinkle('launch')

        self._twinkle(no_draw=True)  # 只是闪

        return missile_launch_pre

    def _plot_region(self, scene):
        '''
        进行预警区域绘制，当未初始化时，先解析输入region并反投影，然后绘制区域在初始化图像上（输入scene可为None）

        已初始化则对输入scene进行绘制

        :param scene:在已初始化时的每帧相机输入图像

        :return:
        已初始化时返回None

        未初始化返回一个布尔类型的ndarray,反映各预警区域的四点反投影点是否在图像内，以供后续重叠区域剔除
        '''
        if self._scene_init:
            for r in self._scene_region.keys():
                # 对于左下方红色和黄色提醒不进行绘制
                if r == "map" or r == "launch":
                    continue
                ips = self._scene_region[r]
                for i in range(4):
                    cv2.line(scene, tuple(ips[i]), tuple(ips[(i + 1) % 4]), (0, 255, 0), 3)
            return None
        else:
            inside = []
            for r in self._region.keys():
                # 格式解析
                ftype, alarm_type, team, loc, height_type = r.split('_')
                if loc not in enemy_case or color2enemy[team] == self._enemy:  # 若区域为敌方限定，则进行判断
                    if ftype == 's' or ftype == 'a':  # 解析是否为需要反投影预警的区域
                        if alarm_type == 'r':
                            # rect type
                            # 左上右下->矩形四点
                            lt = self._region[r][:2].copy()
                            lt[1] = lt[1] - self._real_size[1]  # 左下角原点到左上角原点
                            rd = self._region[r][2:4].copy()
                            rd[1] = rd[1] - self._real_size[1]  # 左下角原点到左上角原点
                            ld = [lt[0], rd[1]]
                            rt = [rd[0], lt[1]]
                            ops = np.float32([lt, rt, rd, ld]).reshape(-1, 2)
                            if height_type == 'a':
                                height = np.ones((ops.shape[0], 1)) * self._region[r][4]
                            if height_type == 'd':
                                height = np.ones((ops.shape[0], 1))
                                height[1:3] *= self._region[r][5]  # r height
                                height[[0, 3]] *= self._region[r][4]  # l height
                        if alarm_type == 'fp':
                            # 四点凸四边形类型
                            ops = np.float32(self._region[r][:8]).reshape(-1, 2)
                            ops[:, 1] -= self._real_size[1]  # 同上，原点变换
                            if height_type == 'a':
                                height = np.ones((ops.shape[0], 1)) * self._region[r][8]
                            if height_type == 'd':
                                height = np.ones((ops.shape[0], 1))
                                height[1:3] *= self._region[r][9]  # r height
                                height[[0, 3]] *= self._region[r][8]  # l height

                        # 反投影
                        ops = np.concatenate([ops, height], axis=1)
                        ips = cv2.projectPoints(ops, self._rvec, self._tvec, self._K_0, self._C_0)[0].astype(
                            int).reshape(-1, 2)
                        # 在初始输入上绘制
                        for p in ips:
                            cv2.circle(self._scene, tuple(p), 10, (0, 255, 0), -1)
                        cv2.fillConvexPoly(self._scene, ips, (0, 255, 0))
                        self._scene_region[r] = ips
                        # 为了统一采用is_inside来判断是否在图像内
                        whole_range = np.array([0, 0, self._size[0] - 1, 0, self._size[0] - 1, self._size[1] - 1, 0,
                                                self._size[1]]).reshape(-1, 2)
                        inside.append(np.array([is_inside(whole_range, p) for p in ips]))

            self._scene_init = True
            return inside

    def check(self, armors, bbox):
        '''
        预警检测
        :param armors:(N,fp+conf+cls+img_no+bbox)
        :param bbox:与armors img_no对应的（详见network类）车辆预测框

        :return:
        whole_alarming 指示是否在任意区域预警

        pred_bbox IoU预测的装甲板定位框 格式（cls+x1+y1+x2+y2) 当没有预测时，为None
        '''
        whole_alarming = False  # 指示是否在任意区域预警
        cache = None  # 当前帧缓存框
        ids = np.array([1, 2, 3, 4, 5, 8, 9, 10, 11, 12])  # id顺序
        f_max = lambda x, y: (x + y + abs(x - y)) // 2
        f_min = lambda x, y: (x + y - abs(x - y)) // 2
        pred_bbox = None  # IoU预测框
        if isinstance(self._cache, np.ndarray) and self._debug:
            # 画上一帧框
            for pre_bbox in self._cache:
                cv2.rectangle(self._every_scene, tuple(pre_bbox[1:3].astype(int)),
                              tuple(pre_bbox[3:].astype(int)), (255, 255, 0), 3)
        if isinstance(armors, np.ndarray) and isinstance(bbox, np.ndarray):
            assert len(armors)
            # 补充预警框
            pred_cls = []
            p_bbox = []  # IoU预测框（装甲板估计后的装甲板框）
            cache_pred = []  # 可能要缓存的当帧预测IoU预测框的原始框
            # 缓存格式 id,x1,y1,x2,y2
            # 根据armors中的img_no来选择对应的车辆预测框
            cache = np.concatenate(
                [armors[:, 9].reshape(-1, 1), np.stack([bbox[int(i)] for i in armors[:, 10]], axis=0)], axis=1)
            cls = armors[:, 9].reshape(-1, 1)
            if isinstance(self._cache, np.ndarray):
                for i in ids:  # 对于每一个id
                    mask = self._cache[:, 0] == i
                    if not (cls == i).any() and mask.any():  # 若没有预测到,且上一帧预测到了
                        # TODO:纠缠型（遮挡物体上有车，导致一直预测为另一辆的框),故先不缓存IoU预测框
                        cache_bbox = self._cache[mask][:, 1:]  # 上一帧bbox
                        # 计算交并比
                        cache_bbox = np.repeat(cache_bbox, len(bbox), axis=0)
                        x1 = f_max(cache_bbox[:, 0], bbox[:, 0])  # 交集左上角x
                        x2 = f_min(cache_bbox[:, 2], bbox[:, 2])  # 交集右下角x
                        y1 = f_max(cache_bbox[:, 1], bbox[:, 1])  # 交集左上角y
                        y2 = f_min(cache_bbox[:, 3], bbox[:, 3])  # 交集右下角y

                        overlap = f_max(np.zeros((x1.shape)), x2 - x1) * f_max(np.zeros((y1.shape)), y2 - y1)
                        union = (cache_bbox[:, 2] - cache_bbox[:, 0]) * (cache_bbox[:, 3] - cache_bbox[:, 1])
                        iou = (overlap / union)

                        if np.max(iou) > self._iou_thre:  # 当最大iou超过阈值值才预测
                            now_bbox = bbox[np.argmax(iou)].copy()  # x1,y1,x2,y2
                            if self._debug:
                                # 被预测的上一帧框（蓝） 预测出的该帧IoU框（红）
                                cv2.rectangle(self._every_scene, tuple(cache_bbox[0, :2].astype(int)),
                                              tuple(cache_bbox[0, 2:].astype(int)), (255, 0, 0), 3)
                                cv2.rectangle(self._every_scene, tuple(now_bbox[:2].astype(int)),
                                              tuple(now_bbox[2:].astype(int)), (0, 0, 255), 3)

                            cache_pred.append(now_bbox.copy())
                            # 装甲板位置估计，详见技术报告
                            w, h = (now_bbox[2:] - now_bbox[:2])
                            now_bbox[2] = w // 3
                            now_bbox[3] = h // 5
                            now_bbox[1] += now_bbox[3] * 3
                            now_bbox[0] += now_bbox[2]

                            if self._debug:
                                # 绘制估计装甲板位置
                                cv2.rectangle(self._every_scene, tuple(now_bbox[:2].astype(int)),
                                              tuple((now_bbox[:2] + now_bbox[2:]).astype(int)), (0, 255, 0), 3)

                            pred_cls.append(np.array(i))
                            p_bbox.append(now_bbox)

            if len(pred_cls):
                # 将cls和四点合并
                pred_bbox = np.concatenate([np.stack(pred_cls, axis=0).reshape(-1, 1), np.stack(p_bbox, axis=0)],
                                           axis=1)
                if self._iou_cache:
                    # 添加IoU预测框至框缓存
                    cache_pred = np.concatenate(
                        [np.stack(pred_cls, axis=0).reshape(-1, 1), np.stack(cache_pred, axis=0)], axis=1)
                    cache = np.concatenate([cache, cache_pred], axis=0)

            # IoU预测框不预警
            points = armors[:, :8].copy()

            # points转换为四点bounding box
            if self._using_bbox:
                x1 = armors[:, 11].reshape(-1, 1)
                y1 = armors[:, 12].reshape(-1, 1)
                x2 = (armors[:, 11] + armors[:, 13]).reshape(-1, 1)
                y2 = (armors[:, 12] + armors[:, 14]).reshape(-1, 1)
                points = np.concatenate([x1, y1, x2, y1, x2, y2, x1, y2], axis=1)
            # 增加仅预测出颜色的敌方预测框
            if isinstance(self._color_bbox, np.ndarray):
                self._color_bbox = self._color_bbox[self._color_bbox[:, 0] == self._enemy]
                if len(self._color_bbox):
                    color_cls = self._color_bbox[:, 0].reshape(-1, 1) - 1  # only detect enemy
                    w = self._color_bbox[:, 3] - self._color_bbox[:, 1]
                    h = self._color_bbox[:, 4] - self._color_bbox[:, 2]
                    self._color_bbox[:, 3] = w // 3  # 长宽变为原来1/5
                    self._color_bbox[:, 4] = h // 5
                    self._color_bbox[:, 2] += self._color_bbox[:, 4] * 3
                    self._color_bbox[:, 1] += self._color_bbox[:, 3]
                    x1 = self._color_bbox[:, 1]
                    y1 = self._color_bbox[:, 2]
                    x2 = x1 + self._color_bbox[:, 3]
                    y2 = y1 + self._color_bbox[:, 4]
                    color_fp = np.stack([x1, y1, x2, y1, x2, y2, x1, y2], axis=1)
                    points = np.concatenate([points, color_fp], axis=0)
                    cls = np.concatenate([cls, color_cls], axis=0)
            points = points.reshape((-1, 4, 2))
            for r in self._scene_region.keys():
                if r == "map" or "launch" in r:  # 提醒区不是反投影预警区
                    continue
                _, _, team, loc, _ = r.split('_')
                # 判断对于各个预测框，是否有点在该区域内
                mask = np.array([[is_inside(self._scene_region[r], p) for p in fp] for fp in points])
                mask = np.sum(mask, axis=1) > 0

                alarm_target = cls[mask]  # 预测出的类（对于颜色 0 is blue -1 is red)
                # 判断是否为敌方
                alarming = np.logical_or(np.logical_and(alarm_target >= 1, alarm_target <= 5) \
                                             if self._enemy else np.logical_and(alarm_target <= 12, alarm_target >= 8),
                                         alarm_target < 1)
                whole_alarming = whole_alarming or alarming.any()
                alarm_target = alarm_target[alarming]

                # 解析从装甲板类别到1-12的标准类别（见_ids定义）
                alarm_target = [self._ids[target] - 1 for target in alarm_target]

                if alarming.any():
                    self._add_twinkle(r)  # 闪烁
                    if loc in ['feipo', 'feipopre', 'gaodipre', 'gaodipre2']:  # 车间通信预警
                        self._touch_api({'task': 2, 'data': [team, loc, alarm_target]})  # （区域颜色，区域名，在区域内目标列表）
                    if self._debug:
                        print("{0} is alarming {1}".format(r, alarm_target[0]))

        if isinstance(cache, np.ndarray):
            for i in ids:
                assert cache[cache[:, 0] == i].reshape(-1, 5).shape[0] <= 1
            self._cache = cache.copy()
        else:
            self._cache = None
        return whole_alarming, pred_bbox

    def plot_alarming(self, alarming, base_alarming):
        '''
        绘制地图预警提醒

        :param alarming:小地图预警
        :param base_alarming:基地预警
        '''
        base_alarming_region = 300 * self._size[0] // 3088, 300 * self._size[0] // 3088
        if alarming:
            self._add_twinkle("map")
        if base_alarming:
            cv2.putText(self._every_scene, "Your home has been stolen!", base_alarming_region, cv2.FONT_HERSHEY_SIMPLEX,
                        5 * self._size[0] // 3088,
                        (0, 0, 255), 3)

    def _add_twinkle(self, region: str):
        '''
        预警闪烁添加函数，和location_alarm那边一样，详见那边
        '''
        if region not in self._twinkle_event.keys():
            self._twinkle_event[region] = self._twinkle_times * 2
        else:
            if self._twinkle_event[region] % 2 == 0:
                self._twinkle_event[region] = self._twinkle_times * 2
            else:
                self._twinkle_event[region] = self._twinkle_times * 2 + 1

    def _twinkle(self, no_draw=False):
        '''
        预警闪烁执行函数，和location_alarm那边一样，详见那边

        :param no_draw:在主程序中如果视角在某一个相机视野，另一个相机是不调用show方法的，这导致其闪烁虽然被添加，
        但是没有闪，故而需要一个函数来使其闪烁次数被消耗
        '''

        for r in self._twinkle_event.keys():
            if self._twinkle_event[r] == 0:
                continue
            if self._twinkle_event[r] % 2 == 0:
                # 闪
                rect = self._scene_region[r]
                if not no_draw:
                    if r == "launch":
                        cv2.fillConvexPoly(self._every_scene, rect, (0, 255, 255))  # 飞镖闪黄灯
                    else:
                        cv2.fillConvexPoly(self._every_scene, rect, (0, 0, 255))
            self._twinkle_event[r] -= 1
