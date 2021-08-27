'''
config.py
该文件存放所有的全局变量数据

对于region的格式定义：
{“{alarm_type}_{shape_type}_{location_type}_{line_assign_type(to line) or height_assign_type(to fp)}”:data(list)}
{alarm_type}
m为map预警，s为反投影预警，a为二者兼可
{shape_type} l or r or fp
{line_assign_type}直线分为左侧和右侧，若为r则只考虑右侧预警，若为l则只考虑左侧预警，若为a则考虑双侧预警
{height_assign_type}fp及r类型预警名称后分为a，d，a代表四点同一高度，d代表两种高度
对于rect(x1,y1),(x1,y2)和（x2,y1),(x2,y2)分别用两种高度
对于fp (x1,y1),(x4,y4)和（x2,y2),(x3,y3)分别用两种高度
{data}
##注意下面坐标均以赛场左下角为坐标原点，向上为y轴，向右为x轴，垂直向上为z轴（笛卡尔坐标系定义）##
line type alarm [x1,y1,x2,y2,dis_thres]
rect type alarm [x1,y1,x2,y2,height1,height2] height为若反投影矩形各点的整体z坐标
fp type alarm [x1,y1,x2,y2,x3,y3,x4,y4,height1,height2] 顺时针,height1为(x1,y1),(x4,y4)的z坐标,height2为（x2,y2),(x3,y3)的z坐标
（四点需要根据上述规则来确定你需要的两侧，第一个点和第二个点在不同侧，也就是不同高度）

区域：
dafu 打符点
base 基地线
diaoshe，R3梯形高地，英雄吊射点
feipopre R4梯形高地去公路区一小段道路
feipo 整段公路区
gaodipre 新加坡
gaodipre2 狗比坡

交龙黑话：
新加坡：指21赛季环高靠近荒地出口的坡
狗比坡：指21赛季环高靠近飞坡点的坡
'''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
|                       重要参数调整区域                            |
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
enemy: int = 0  # 0:red, 1:blue
BO:int = 1 # Max BO
# 主程序专用
battle_mode: bool = True # 比赛模式，一般开着便可，会进行logging的记录,不开，会打印fps,logging只记录fps
usb: str = '/dev/ttyUSB0'
split = False # 是否对反投影重叠区域进行去重
##### 主程序测试参 ######## 当测试总开关被设置为False，将会将所有advanced测试参数设置默认非测试参数
test = True
######################################

region = \
{

    'm_l_red_base_r': [5.500, 9.700, 5.500, 5.300, 0.500], # red base only for enemy 1
    'm_l_blue_base_l':[22.500,9.700,22.500,5.300,0.500], # blue base only for enemy 0
    'a_fp_red_diaoshe_a': [8.682,13.906,6.101,10.117,4.536,10.156,4.575,14.023, 0.45, 0.], # enemy case,only alarm
    'a_fp_blue_diaoshe_a':[23.464,4.844,23.425,0.977,19.318,1.094,21.899,4.883,0.45,0.],
    'm_r_red_dafu_a': [7.718, 2.532, 9.130, 1.315, 0.945, 0.], # enemy case, only alarm
    'm_r_red_feipopre_a': [2.855, 3.516, 4.184, 0.000, 0.4, 0.], # only enemy case, only alarm
    'a_r_red_feipo_a': [4.536, 1.016, 12.318, 0.156, 0.3, 0.], # only enemy case, send and alarm : send to 1,3,4,5 TASK 2
    'm_r_blue_dafu_a': [18.870,13.685,20.282,12.468, 0.945, 0.],
    'm_r_blue_feipopre_a': [23.816, 15.000, 25.145, 11.484, 0.4, 0.],
    'a_r_blue_feipo_a': [15.682, 14.844, 23.464, 13.984, 0.3, 0.],
    'm_fp_red_gaodipre_d':[10.421,7.183,11.614,5.357,10.421,4.554,8.960,6.721,0.6,0.], # every case, send and alarm : send to 1,3,4,5 TASK 2
    'm_fp_blue_gaodipre_d':[17.579,10.446,19.040,8.279,17.579,7.817,16.386,9.643,0.,0.6], # xinjiapo
    'm_fp_red_gaodipre2_a': [13.531, 13.906, 11.497, 11.211, 10.559, 11.836, 12.318, 14.492, 0., 0.],# every case, send and alarm : send to 1,3,4,5 TASK 2
    'm_fp_blue_gaodipre2_a':[17.441,3.164,15.682,0.508,14.469,1.094,16.503,3.789,0.,0.], # goubipo
    's_fp_red_missilelaunch2_d': [0.365,11.761,1.266,11.810,1.266,9.789,0.243,9.862, 4, 2.4],
    's_fp_red_missilelaunch1_d': [1.290,11.323,0.609,11.347,0.682,10.300,1.290,10.252, 1.1, 0.4],
    's_fp_blue_missilelaunch2_d':[26.734,5.211,27.757,5.138,27.635,3.239,26.734,3.190,2.4, 4],
    's_fp_blue_missilelaunch1_d': [27.318,4.700,26.710,4.748,26.710,3.677,27.391,3.653, 0.4, 1.1]
}
# 经过转换过的区域定义 (28.,15.) -> (12.,6.) 转换函数见 tools/generate_region.py
test_region = \
    {
        'm_l_red_base_r': [2.357, 3.880, 2.357, 2.120, 0.500],
        'm_l_blue_base_l': [9.643, 3.880, 9.643, 2.120, 0.500],
        'a_fp_red_diaoshe_a': [3.721, 5.562, 2.615, 4.047, 1.944, 4.062, 1.961, 5.609, 0., 0.],
        'a_fp_blue_diaoshe_a': [10.056, 1.938, 10.039, 0.391, 8.279, 0.438, 9.385, 1.953, 0., 0.],
        'm_r_red_dafu_a': [3.218, 1.000, 3.771, 0.500, 0., 0.],
        'm_r_red_feipopre_a': [1.224, 1.406, 1.793, 0.000, 0., 0.],
        'a_r_red_feipo_a': [1.944, 0.406, 5.279, 0.062, 0., 0.],
        'm_r_blue_dafu_a': [8.229, 5.500, 8.782, 5.000, 0., 0.],
        'm_r_blue_feipopre_a': [10.207, 6.000, 10.776, 4.594, 0., 0.],
        'a_r_blue_feipo_a': [6.721, 5.938, 10.056, 5.594, 0., 0.],
        'm_fp_red_gaodipre_d': [4.391, 2.891, 5.011, 2.047, 4.509, 1.797, 3.922, 2.641, 0., 0.],
        'm_fp_blue_gaodipre_d': [7.491, 4.203, 8.078, 3.359, 7.609, 3.109, 6.989, 3.953, 0., 0.],
        's_fp_red_gaodipre2_a': [5.799, 5.562, 4.927, 4.484, 4.525, 4.734, 5.279, 5.797, 0., 0.],
        's_fp_blue_gaodipre2_a': [7.475, 1.266, 6.721, 0.203, 6.201, 0.438, 7.073, 1.516, 0., 0.],
        's_fp_red_missilelaunch2_d': [0.156, 4.704, 0.543, 4.724, 0.543, 3.916, 0.104, 3.945, 0.5, 0.],
        's_fp_red_missilelaunch1_d': [0.553, 4.529, 0.261, 4.539, 0.292, 4.120, 0.553, 4.101, 0.5, 0.],
        's_fp_blue_missilelaunch2_d': [11.457, 2.084, 11.896, 2.055, 11.844, 1.296, 11.457, 1.276, 0., 0.5],
        's_fp_blue_missilelaunch1_d': [11.708, 1.880, 11.447, 1.899, 11.447, 1.471, 11.739, 1.461, 0., 0.5],
    }

##### advanced 测试参 #####
if test:
    home_test = False  # 是否是在家里测试(用缩小版尺寸)
    using_dq = True # 使用record的点云来代替实际的radar节点输入
    PC_RECORD_SAVE_DIR = "demo_resource/demo_pc.pkl" # 点云存放位置
    real_size = (28.,15.)
    main_region = region
    debug = True
    using_video = True  # 测试是否用视频
    VIDEO_PATH = "demo_resource/two_cam"
    THIRD_VIDEO_PATH = "demo_resource/third_cam.mp4"
else: # default
    home_test = False
    using_dq = False
    PC_RECORD_SAVE_DIR = None
    real_size = (28.,15.)
    main_region = region
    debug = False
    using_video = False
    VIDEO_PATH = None
    THIRD_VIDEO_PATH = None
####################

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
|                       重要参数调整结束                            |
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

color2enemy = {"red":0,"blue":1}
enemy2color = ['red','blue']

enemy_case = ["diaoshe", 'dafu', 'feipopre', 'feipo', 'missilelaunch1', "missilelaunch2"]  # 这些区域在预警时只考虑敌方的该区域

armor_list = ['R1','R2','R3','R4','R5','B1','B2','B3','B4','B5'] # 雷达站实际考虑的各个装甲板类

unit_list = ['R1','R2','R3','R4','R5','RG','RO','RB','B1','B2','B3','B4','B5','BG','BO','BB'] # 赛场上各个目标，主要用于HP显示

state2color = {True: "#00FF00", False: "#FF0000"} # 雷达和位姿估计提示中布尔类型和颜色查找字典

# 小地图图片路径
MAP_PATH = "demo_resource/map2.jpg"
# 小地图设定大小
map_size = (716,384)
# UI中主视频源初始图像路径
INIT_FRAME_PATH = "demo_resource/demo_pic.jpg"
# 视频保存路径
VIDEO_SAVE_DIR = "record_competition"
# 输出logging保存路径
OUTPUT_LOGGING_DIR = "output_log"
# 特定logging保存路径，分为非battle_mode和battle_mode
SPECIFIC_LOGGING_BATTLE_MODE_DIR = "battle_log"
SPECIFIC_LOGGING_NON_BATTLE_MODE_DIR = "log"
# [demo only] network "weights" dir
DEMO_PKL_DIR = "demo_resource/demo_infer.pkl"

# 裁判系统发送编号定义
loc2code = \
    {
        'dart': 0,
        'feipo': 1,
        'feipopre': 2,
        "gaodipre": 3,  # xinjiapo
        "gaodipre2": 4,  # goubipo
        'base': 5,

    }
loc2car = {
    'dart': [7],
    'feipopre': [1],
    'feipo': [1, 3, 4, 5],
    'gaodipre': [1, 3, 4, 5],
    'gaodipre2': [1, 3, 4, 5],
    'base': [1, 2, 3, 4, 5]
}

# 位姿估计常量
LOCATION_SAVE_DIR = "pose_save"
location_targets = {
    'home_test': # 家里测试，填自定义类似于赛场目标的空间位置
    {
        'red_base': [10.872, -15. + 2.422, 1.261],
        'blue_outpost': [12 - 5.1, -1.2, 0],
        'red_outpost': [12 - 5.4, -4.8, 0],
        'blue_base': [12 - 0.9, -2.4, 0],
        'r_rt': [12 - 2.7, -3.6, 0],  # r0 right_top
        'r_lt': [12 - 1.8, -2.4, 0],  # r0 left_top
        'b_rt': [19.200, -9.272 + 0.660, 0.120 + 0.495],  # b0 right_top
        'b_lt': [19.200, -9.272, 0.120 + 0.495]  # b0 right_top
    },

    'game': # 按照官方手册填入
        {
            'red_base': [1.760, -15. + 7.539, 0.200 + 0.920],  # red base
            'blue_outpost': [16.776, -15. + 12.565, 1.760],  # blue outpost
            'red_outpost': [11.176, -15. + 2.435, 1.760],  # red outpost
            'blue_base': [26.162, -15. + 7.539, 0.200 + 0.920],  # blue base
            'r_rt': [8.805, -5.728 - 0.660, 0.120 + 0.495],  # r0 right_top
            'r_lt': [8.805, -5.728, 0.120 + 0.495],  # r0 left_top
            'b_rt': [19.200, -9.272 + 0.660, 0.120 + 0.495],  # b0 right_top
            'b_lt': [19.200, -9.272, 0.120 + 0.495]  # b0 left_top
        }
}


# 相机驱动参数
# MindVision相机唯一序列号
camera_match_list = \
    ['049001810123',  # right camera
     '041031120567',  # left camera
     '027041310080']  # top camera

# 调参界面位置参数,由于多进程设置，调参界面会同时产生，为防止误关，将同时产生的两个界面分离显示
preview_location = [
    (100, 100), (940, 100)
]  # 左右相机使用第一个位置，上相机使用第二个位置

# 相机默认参数文件位置 .Config
CAMERA_CONFIG_DIR = "Camera"
# 相机运行中缓存参数文件位置
CAMERA_CONFIG_SAVE_DIR = "Camera/in_battle_config"
# 相机标定文件位置 .yaml
CAMERA_YAML_PATH = "Camerainfo"

# 雷达驱动参数
PC_STORE_DIR = "point_record" # 录制点云保存位置
LIDAR_TOPIC_NAME = "/livox/lidar" # 雷达PointCloud节点名称

# 反导时间参数
two_stage_time = 18.  # 反导第二阶段最大持续时间

# 多进程相机录像保存路径
THIRD_CAMERA_SAVE_DIR = "record_competition_high_focus"

# 屏幕大小
win_size = (1920,1080)


