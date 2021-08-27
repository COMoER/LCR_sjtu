'''
裁判系统统合类
'''
import time
import numpy as np
from serial_package import offical_Judge_Handler, Game_data_define
import queue

from radar_class.config import enemy,BO
###### 采自官方demo ###########
ind = 0 # 发送id序号（0-4）
Id_red = 1
Id_blue = 101

buffercnt = 0
buffer = [0]
buffer *= 1000
cmdID = 0
indecode = 0

def ControlLoop_red():
    '''
    循环红色车辆id
    '''
    global Id_red
    if Id_red == 5:
        Id_red = 1
    else:
        Id_red = Id_red + 1


def ControlLoop_blue():
    '''
    循环蓝色车辆id
    '''
    global Id_blue
    if Id_blue == 105:
        Id_blue = 101
    else:
        Id_blue = Id_blue + 1


def read(ser):
    global buffercnt
    buffercnt = 0
    global buffer
    global cmdID
    global indecode
    # TODO:qt thread

    while True:
        s = ser.read(1)
        s = int().from_bytes(s, 'big')
        # doc.write('s: '+str(s)+'        ')

        if buffercnt > 50:
            buffercnt = 0

        # print(buffercnt)
        buffer[buffercnt] = s
        # doc.write('buffercnt: '+str(buffercnt)+'        ')
        # doc.write('buffer: '+str(buffer[buffercnt])+'\n')
        # print(hex(buffer[buffercnt]))

        if buffercnt == 0:
            if buffer[buffercnt] != 0xa5:
                buffercnt = 0
                continue

        if buffercnt == 5:
            if offical_Judge_Handler.myVerify_CRC8_Check_Sum(id(buffer), 5) == 0:
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 7:
            cmdID = (0x0000 | buffer[5]) | (buffer[6] << 8)
            # print("cmdID")
            # print(cmdID)

        if buffercnt == 10 and cmdID == 0x0002:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                Referee_Game_Result()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 20 and cmdID == 0x0001:

            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 20):
                # 比赛阶段信息
                UART_passer.Referee_Update_GameData()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 41 and cmdID == 0x0003:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 41):
                # 各车血量
                UART_passer.Referee_Robot_HP()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 12 and cmdID == 0x0004:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 12):
                Referee_dart_status()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 13 and cmdID == 0x0101:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 13):
                Referee_event_data()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 13 and cmdID == 0x0102:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 13):
                Refree_supply_projectile_action()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 11 and cmdID == 0x0104:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 11):
                Refree_Warning()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 10 and cmdID == 0x0105:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                Refree_dart_remaining_time()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 17 and cmdID == 0x301:  # 2bite数据
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 17):
                # 比赛阶段信息
                UART_passer.Receive_Robot_Data()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 25 and cmdID == 0x202:  # 雷达没有
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 25):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 25 and cmdID == 0x203:  # 雷达没有
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 25):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 27 and cmdID == 0x201:  # 雷达没有
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 27):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 10 and cmdID == 0x204:  # 雷达没有
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 10 and cmdID == 0x206:  # 雷达没有
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 13 and cmdID == 0x209:  # 雷达没有
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 13):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 16 and cmdID == 0x0301:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 16):
                # Refree_map_stop()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 24 and cmdID == 0x0303:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 24):
                # 云台手通信
                Refree_Arial_Message()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        buffercnt += 1


Game_state = Game_data_define.game_state()
Game_result = Game_data_define.game_result()
Game_robot_HP = Game_data_define.game_robot_HP()
Game_dart_status = Game_data_define.dart_status()
Game_event_data = Game_data_define.event_data()
Game_supply_projectile_action = Game_data_define.supply_projectile_action()
Game_refree_warning = Game_data_define.refree_warning()
Game_dart_remaining_time = Game_data_define.dart_remaining_time()

################################################

class UART_passer(object):
    '''
    convert the Judge System message
    自定义裁判系统统合类
    '''
    # message box controller
    _bytes2int = lambda x:(0x0000 | x[0]) | (x[1] << 8)
    _hp_up = np.array([100,150,200,250,300,350,400,450,500]) # 各个升级血量阶段
    _init_hp = np.ones(10,dtype = int)*500 # initialize all car units' hp as 500
    _last_hp = _init_hp.copy()
    _HP = np.ones(16,dtype = int)*500 # to score the received hp message, initialize all as 500
    _max_hp = _init_hp.copy() # store the hp maximum hp of each car
    _set_max_flag = False # When the game starts, set their maximum hp in the beginning of the game via receiving the first hp message.

    # location controller
    _robot_location = np.zeros((5,2),dtype=np.float32) # the enemy car location received from the alarm class

    # alarming event queue with priority
    _queue = queue.PriorityQueue(-1)
    # Game State controller
    _BO = 0
    _stage = ["NOT START", "PREPARING", "CHECKING", "5S", "PLAYING", "END"]
    _Now_Stage = 0
    _Game_Start_Flag = False
    _Game_End_Flag = False
    Remain_time = 0
    # Arial control flag 云台手控制置位符
    change_view = False
    anti_dart = False
    open_base = False
    getposition = False
    _HP_thres = 0 # 发送预警的血量阈值，即预警区域中血量最高的车辆的血量低于该阈值则不发送预警
    _prevent_time = [2.,2.,2.,2.,2.,2.] # sleep time 预警发送沉默时间，若距离上次发送小于该阈值则不发送
    _event_prevent = np.zeros(6) # 距离时间记录
    loop_send = 0 # 在一次小地图车辆id循环中发送的次数

    @staticmethod
    def _judge_max_hp(HP):
        '''
        血量最大值变化判断逻辑，by 何若坤，只适用于21赛季规则
        '''
        mask_zero = UART_passer._last_hp > 0 # 血量为0不判断
        focus_hp = HP[[0,1,2,3,4,8,9,10,11,12]] # 只关心这些位置的血量上限
        # 工程车血量上限不变，不判断
        mask_engineer = np.array([True]*10)
        mask_engineer[[1,6]] = False
        mask_engineer = np.logical_and(mask_zero,mask_engineer)
        # 若血量增加在30到80，则为一级上升（50）
        mask_level1 = np.logical_and(focus_hp-UART_passer._last_hp>30,focus_hp-UART_passer._last_hp<=80)
        # 若血量增加在80以上，则为二级上升（100）
        mask_level2 = focus_hp-UART_passer._last_hp > 80
        UART_passer._max_hp[np.logical_and(mask_level1,mask_engineer)] += 50
        UART_passer._max_hp[np.logical_and(mask_level2,mask_engineer)] += 100
        # 如果有一次上限改变没检测到，使得当前血量大于上限，则调整至相应的上限
        mask_still = np.logical_and(focus_hp > UART_passer._max_hp,mask_engineer)
        for i in np.where(mask_still)[0]:
            UART_passer._max_hp[i] = np.min(UART_passer._hp_up[UART_passer._hp_up > focus_hp[i]])

        UART_passer._last_hp = focus_hp.copy()

    def __init__(self):
        pass

    @staticmethod
    def push_loc(location):
        '''
        放入当前帧位置
        '''
        UART_passer._robot_location = np.float32(location)
    @staticmethod
    def get_position():
        '''
        得到当前帧位置
        '''
        return UART_passer._robot_location

    @staticmethod
    def _send_check(code,alarm_target):
        '''
        预警发送判断逻辑

        :param code:发送代号
        :param alarm_target:落在预警区域的车辆列表(0-9)

        :return: True if the alarming could be sent
        '''
        if code < 1: # 飞镖预警不需要判断血量
            return True
        else:
            HP = UART_passer._HP[[0,1,2,3,4,8,9,10,11,12]]
            alarm_tag = np.array(alarm_target,dtype = int)
            alarm_tag = alarm_tag[alarm_tag < 10].tolist()
            if (HP[alarm_tag] > UART_passer._HP_thres).any():
                return True
            else:
                # If all the robot in the alarming region are death, then don't send the alarming.
                return False

    @staticmethod
    def push(code,send_target,alarm_target,team):
        '''
        将一个预警push进预警队列，以code作为优先级，code越小，优先级越高
        '''
        if UART_passer._send_check(code,alarm_target):
            UART_passer._queue.put((code,send_target,alarm_target,team))
    @staticmethod
    def pop():
        '''
        pop出一个预警信息
        '''
        if not UART_passer._queue.empty():
            code,send_targets,alarm_targets,team = UART_passer._queue.get()
            t = time.time()
            # 判断预警间隔
            if t - UART_passer._event_prevent[code] > UART_passer._prevent_time[code]:
                UART_passer._event_prevent[code] = t # reset
                return True,code,send_targets,alarm_targets,team
            else:
                return False,None,None,None,None
        else:
            return False,None,None,None,None
    @staticmethod
    def get_message(hp_scene):
        '''
        基于裁判系统类保存的血量及比赛阶段信息，更新hp信息框
        '''
        hp_scene.refresh()
        hp_scene.update(UART_passer._HP,UART_passer._max_hp)
        hp_scene.update_stage(UART_passer._stage[UART_passer._Now_Stage],UART_passer.Remain_time,UART_passer._BO+1,BO)

    @staticmethod
    def Referee_Update_GameData():
        # If the game state is from starting to ending or from ending to starting, set stage change flag
        # From Stage "preparing" to "15s" or "5s" or "playing", any of them will enable this(GAME START)
        if UART_passer._Now_Stage < 2 and ((buffer[7] >> 4) == 2 or (buffer[7] >> 4) == 3 or
                                           (buffer[7] >> 4) == 4):
            # ONE GAME START
            UART_passer._Game_Start_Flag = True
            UART_passer._set_max_flag = True
        if UART_passer._Now_Stage < 5 and (buffer[7] >> 4) == 5:
            # ONE GAME END
            UART_passer._Game_End_Flag = True
            UART_passer._max_hp = UART_passer._init_hp.copy()
        UART_passer._Now_Stage = buffer[7] >> 4

        UART_passer.Remain_time = (0x0000 | buffer[8]) | (buffer[9] << 8)
    @staticmethod
    def Referee_Robot_HP():
        # 1 2 3 4 5 guard outpost base 血量顺序
        UART_passer._HP = np.array([UART_passer._bytes2int((buffer[i*2-1],buffer[i*2])) for i in range(4,20)],dtype = int)
        if UART_passer._set_max_flag:
            # 比赛开始时，根据读取血量设置最大血量
            UART_passer._max_hp = UART_passer._HP[[0,1,2,3,4,8,9,10,11,12]]
            UART_passer._set_max_flag = False
        else:
            UART_passer._judge_max_hp(UART_passer._HP)

    @staticmethod
    def One_compete_end():
        # check the state change flag
        if UART_passer._Game_End_Flag:
            # reset
            UART_passer._Game_End_Flag = False
            UART_passer._BO += 1
            return True,UART_passer._BO - BO
        else:
            return False,-1

    @staticmethod
    def One_compete_start():
        # check the state change flag
        if UART_passer._Game_Start_Flag:
            # reset
            UART_passer._Game_Start_Flag = False
            return True
        else:
            return False

    @staticmethod
    def Receive_Robot_Data():
        '''
        between car receiver api, now no use
        '''
        if (0x0000 | buffer[7]) | (buffer[8] << 8) == 0x0200:
            print("received")

def Refree_Arial_Message():
    '''
    Arial controller
    radar keyboard
    云台手车间通信按下按键信息接收
    L transpose view
    U anti dart
    O open base
    to be continued...
    '''
    key = buffer[19]
    if key == ord('L')&0xFF: # 设置视角切换置位符
        UART_passer.change_view = True
    if key == ord('U')&0xFF: # 设置反导进入第二阶段置位符
        UART_passer.anti_dart = True
    if key == ord('O')&0xFF: # 设置多进程相机操作状态
        UART_passer.open_base = not UART_passer.open_base

############# 官方demo的函数 ###########
def Judge_Refresh_Result():
    print("Judge_Refresh_Result")


def Referee_Game_Result():
    # print("Referee_Game_Result")
    Game_result.winner = buffer[7]
    # print("The winner is {0}".format(Game_result.winner))

def Referee_dart_status():
    # print("Referee_dart_status")
    Game_dart_status.dart_belong = buffer[7]
    Game_dart_status.stage_remaining_time = [buffer[8], buffer[9]]


def Referee_event_data():
    # print("Referee_event_data")
    Game_event_data.event_type = [buffer[7], buffer[8], buffer[9], buffer[10]]

    # doc.write('Referee_event_data' + '\n')


def Refree_supply_projectile_action():
    # print("Refree_supply_projectile_action")
    Game_supply_projectile_action.supply_projectile_id = buffer[7]
    Game_supply_projectile_action.supply_robot_id = buffer[8]
    Game_supply_projectile_action.supply_projectile_step = buffer[9]
    Game_supply_projectile_action.supply_projectile_num = buffer[10]


def Refree_Warning():
    # print("Refree_Warning")
    Game_refree_warning.level = buffer[7]
    Game_refree_warning.foul_robot_id = buffer[8]


def Refree_dart_remaining_time():
    # print("Refree_dart_remaining_time")
    Game_dart_remaining_time.time = buffer[8]
    # doc.write('Refree_dart_remaining_time' + '\n')

####################################
        
def Referee_Transmit_BetweenCar(dataID, ReceiverId, data, ser):
    '''
    雷达站发送车间通信包函数
    '''
    buffer = [0]
    buffer = buffer * 200

    buffer[0] = 0xA5  # 数据帧起始字节，固定值为 0xA5
    buffer[1] = 10  # 数据帧中 data 的长度,占两个字节
    buffer[2] = 0
    buffer[3] = 0  # 包序号
    buffer[4] = offical_Judge_Handler.myGet_CRC8_Check_Sum(id(buffer), 5 - 1, 0xff)  # 帧头 CRC8 校验
    buffer[5] = 0x01
    buffer[6] = 0x03
    # 自定义内容ID
    buffer[7] = dataID & 0x00ff
    buffer[8] = (dataID & 0xff00) >> 8
    # 发自雷达站
    if enemy:
        buffer[9] = 9
    else:
        buffer[9] = 109
    buffer[10] = 0
    buffer[11] = ReceiverId
    buffer[12] = 0
    # 自定义内容数据段
    buffer[13] = data[0]
    buffer[14] = data[1]
    buffer[15] = data[2]
    buffer[16] = data[3]

    offical_Judge_Handler.Append_CRC16_Check_Sum(id(buffer), 10 + 9)  # 等价的

    buffer_tmp_array = [0]
    buffer_tmp_array *= 9 + 10

    for i in range(9 + 10):
        buffer_tmp_array[i] = buffer[i]
    ser.write(bytearray(buffer_tmp_array))


def Referee_Transmit_Map(cmdID, datalength, targetId, x, y, ser):
    '''
    小地图包

    x，y采用np.float32转换为float32格式
    '''
    buffer = [0]
    buffer = buffer * 200

    buffer[0] = 0xA5  # 数据帧起始字节，固定值为 0xA5
    buffer[1] = (datalength) & 0x00ff  # 数据帧中 data 的长度,占两个字节
    buffer[2] = ((datalength) & 0xff00) >> 8
    buffer[3] = 0  # 包序号
    buffer[4] = offical_Judge_Handler.myGet_CRC8_Check_Sum(id(buffer), 5 - 1, 0xff)  # 帧头 CRC8 校验
    buffer[5] = cmdID & 0x00ff
    buffer[6] = (cmdID & 0xff00) >> 8

    buffer[7] = targetId
    buffer[8] = 0
    buffer[9] = bytes(x)[0]
    buffer[10] = bytes(x)[1]
    buffer[11] = bytes(x)[2]
    buffer[12] = bytes(x)[3]
    buffer[13] = bytes(y)[0]
    buffer[14] = bytes(y)[1]
    buffer[15] = bytes(y)[2]
    buffer[16] = bytes(y)[3]
    buffer[17:20] = [0] * 4 ## 朝向，直接赋0，协议bug，不加这一项无效

    offical_Judge_Handler.Append_CRC16_Check_Sum(id(buffer), datalength + 9)  # 等价的

    buffer_tmp_array = [0]
    buffer_tmp_array *= 9 + datalength

    for i in range(9 + datalength):
        buffer_tmp_array[i] = buffer[i]
    ser.write(bytearray(buffer_tmp_array))



def Robot_Data_Transmit_Map(ser):
    global ind
    location = UART_passer.get_position()[ind]

    # 检查该位置是否被置零
    if np.isclose(location,0).all():
        flag = False
    else:
        flag = True
    x,y = location

    if enemy == 0:
        if flag: # 置零便不发送处理
            # 小地图协议，车辆id,float32类型x坐标，float32类型y坐标，串口对象
            Referee_Transmit_Map(0x0305, 14, Id_red, np.float32(x), np.float32(y), ser)
            time.sleep(0.1)
        ControlLoop_red() # 循环红色车编号 1-5
    else:
        if flag:
            Referee_Transmit_Map(0x0305, 14, Id_blue, np.float32(x), np.float32(y), ser)
            time.sleep(0.1)
        ControlLoop_blue() # 循环红色车编号 101-105
    if flag:
        UART_passer.loop_send += 1 # 在一轮循环中发送次数
    if ind == 4:
        if  UART_passer.loop_send == 0: # 若一轮循环结束，没有发送一次，则sleep 0.1s来保证不占主线程
            time.sleep(0.1)
        UART_passer.loop_send = 0
    ind = (ind + 1) % 5

def write(ser):
    # 写线程
    while True:
        Robot_Data_Transmit_Map(ser)
        flag,code,send_target,alarm_targets,team = UART_passer.pop()
        if flag:
            for t in send_target:
                if enemy == 0: # 车间通信目标转换为蓝方
                    t += 100
                dataID = 0x0200 + (code & 0xFF)

                print("send code {0} to {1} in which is {2}".format(code,t,alarm_targets))
                if code == 0: # anti dart message
                    send_t = UART_passer.Remain_time
                    Referee_Transmit_BetweenCar(dataID, t, [send_t&0xff, (send_t&0xff00)>>8, 0, 0], ser) # 包数据端，发送反导信息时，比赛剩余时间
                if code > 0:
                    target_code = 0
                    for target in alarm_targets:
                        if target <= 9:
                            target_code += 1<<(target%5) # 基于二进制编码，从右到左各位若为1则表示该位置车辆在预警区域内
                    if code in [3,4]: # 对于两个高地坡，需要发送该坡为己方坡还是敌方坡 0 is enemy, 1 is ours
                        Referee_Transmit_BetweenCar(dataID, t, [target_code, (team!=enemy)&0xFF, 0, 0], ser)
                    else:
                        Referee_Transmit_BetweenCar(dataID, t, [target_code, 0, 0, 0], ser)
                time.sleep(0.1)