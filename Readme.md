## LCR_sjtu

上海交通大学交龙战队2021赛季雷达站程序开源

### 一、功能介绍

本程序为除神经网络外的雷达站项目整体框架。基于MindVision相机和Livox MID70激光雷达传感器。实现了<b>雷达点云处理生成深度图</b>，相机图像采集，对其经过神经网络处理后的预测框进行一系列<b>后处理</b>，得到敌方和我方车辆世界坐标系定位信息并基于这些信息作出<b>预警</b>。对于神经网络，我们第一层网络完全基于原始yolov5 4.0版本项目，可于项目原地址进行下载和使用，我们会在release中提供我们所训练的网络权重。

| 开发人员 | 工作                                 | 微信号         |
| -------- | ------------------------------------ | -------------- |
| 郑煜     | 2021赛季雷达站负责人                 | comoersjtu     |
| 黄弘骏   | 2020赛季雷达站负责人（老一辈雷达人） | HUANGHONGJUN-x |

<b>欢迎交流以及告诉我们bug，让我们一起把雷达站做成真正的战术AI</b>

### 二、效果展示

- UI界面

<img src="doc_imgs\missile.png" alt="missile" style="zoom:67%;" />

- 飞镖预警

<img src="doc_imgs\UI.jpg" alt="missile" style="zoom:67%;" />

- 我们demo的测试视频放在了b站上[视频演示](https://www.bilibili.com/video/BV1FM4y1579H/)，以便大家熟悉我们雷达站测试的使用，附有讲解
- 由于加入了两个相机，以及预测经过了两层yolov5s量级的网络，我们雷达站帧率在5-10帧左右，对于小地图通信频率是适应的，但对于更高的实时性要求便有逊色，待后续进行优化。

### 三、环境配置

| 软件环境     | 硬件环境                                          |
| ------------ | ------------------------------------------------- |
| Ubuntu 18.04 | NVIDIA GeForce GTX 1080Ti                         |
| ROS Melodic  | MV-SUA630C-T + <br>手动变焦镜头4-18mm 3mp 1/1.8'' |
| Anaconda     | MV-UBS31GM  <br>+ 长焦镜头 35mm 5mp 2/3''         |
|              | Livox Mid70                                       |
|              | USB转TTL                                          |

MindVision相机驱动下载安装[迈德威视](http://www.mindvision.com.cn/rjxz/list_12.aspx?lcid=138)，[开发demo例程](http://www.mindvision.com.cn/rjxz/list_12.aspx?lcid=139)

Anaconda所需要的库请使用下列命令进行安装

```shell
pip install -r requirements.txt
```

- 以上库只包含运行demo需要的文件，当您添加您自己的神经网络时，请根据<b>radar_class/network.py</b>的提示进行类封装,并在main_l.py中的各TODO项中进行修改

ROS Melodic安装参见[ROS Melodic](https://blog.csdn.net/haiyinshushe/article/details/84256137)

- 安装完成后，请安装Livox官方ROS驱动[Livox ROS](https://github.com/Livox-SDK/livox_ros_driver)

- 启动驱动，可参照我们提供的<b>scripts/start.sh</b>进行快速启动

### 四、程序使用

#### demo运行方式

- 环境安装后，下载release中的demo_resource压缩包解压，放置于该项目根目录下，无需调整<b>radar_class/config.py</b>中任何参数，即可输入以下命令运行

    ```shell
    python main_l.py
    ```

- 当然如果你安装了ROS环境并想采用你自己的神经网络运行该程序（即在比赛中运行），请修改<b>radar_class/config.py</b>中的参数，各个参数我们有详细注释，并替换神经网络类文件
- 考虑上赛场上自启动的需求，我们在开发中同样添加了自启动脚本```test.bash```  ```main_l.sh``` ```start.sh```请参照文件注释进行使用

| 脚本      | 用处                                                         |
| --------- | ------------------------------------------------------------ |
| test.bash | 会自动创建两个窗口，一个运行start.sh以开启雷达驱动，一个运行main_l.sh运行雷达主程序 |
| main_l.sh | 先加载ROS环境（可选），再加载conda环境，再启动main_l.py脚本  |
| start.sh  | 包含加载devel/start.bash以及roslaunch两个过程                |

#### 相机配置文件

配置文件中camera后编号即对应程序中相机编号（camera_type) 0为右相机，1为左相机，2为上相机

##### Config文件

该文件由MindVison Windows demo程序生成

##### yaml文件

yaml文件中保存相机标定参数,请标定后填入该文件（K_0为内参，C_0为畸变系数，E_0为雷达到相机外参）

### 五、文件目录

```
LCR_sjtu
│  .gitignore 
│  Demo_v4.py # PyQt5产生的UI设计python实现
│  LICENSE
│  mainEntry.py # 自定义UI类
│  main_l.py # 主程序
│  main_l.sh # 三个自启动脚本
│  start.sh
│  test.bash
│  Readme.md
│  requirements.txt # pip安装环境
│  UART.py # 裁判系统驱动
├─Camera # 相机参数
│      camera_0.Config
│      camera_1.Config
│      camera_2.Config
│      
├─Camerainfo # 相机标定参数
│      camera0.yaml
│      camera1.yaml
│      camera2.yaml
│      
├─demo_resource # 运行demo资源
│  │  demo_infer.pkl # 保存的神经网络预测文件
│  │  demo_pc.pkl # 保存的点云文件
│  │  demo_pic.jpg # 示例背景图
│  │  map2.jpg # 示例小地图
│  │  third_cam.mp4 # 示例上相机视频
│  │  
│  └─two_cam # 示例左右相机视频
│          1.mp4
│          2.mp4
│      
├─radar_class # 主要类
│     camera.py # 相机驱动类
│     common.py # 各类常用函数,包括绘制，装甲板去重
│     config.py # 配置文件
│     Lidar.py # 雷达驱动
│     location.py # 位姿估计
│     location_alarm.py # 位置预警类
│     missile_detect.py # 飞镖预警类
│     multiprocess_camera.py # 多进程相机类
│     network.py # 示例神经网络类
│     reproject.py # 反投影预警类
│     ui.py # hp及比赛阶段UI类
│          
├─serial_package # 官方的裁判系统驱动
│     Game_data_define.py
│     offical_Judge_Handler.py
│     init.py
│          
├─tools 
│      Demo_v4.ui # QtUI原始文件
│      generate_region.py # 产生感兴趣区域
│      
└─_sdk # mindvision相机驱动文件
       mvsdk.py
       init.py
```

### 六、程序流程图

<img src="doc_imgs\main_process.png" alt="image-20210827143234164" style="zoom:67%;" />

### 七、雷达定位原理

具体原理详见技术报告[RM2021-上海交通大学-云汉交龙战队-雷达站算法部分开源](https://bbs.robomaster.com/forum.php?mod=viewthread&tid=12239)，这里为简述

#### 神经网络预测

神经网络预测得到车辆预测框和装甲板预测框，进行装甲板去重，得到每个车辆id唯一对应的装甲板框，作为车辆图像定位框

#### 雷达深度图

基于以下公式，基于雷达和相机标定关系，将点云投影到相机平面，形成深度图。并使用队列进行点云在一段时间内的积分(更新公式如下，即取二者最小值）。

<img src="https://latex.codecogs.com/svg.image?z_{c}&space;\left[&space;\begin{matrix}&space;u&space;&space;\\&space;v&space;\\&space;1&space;&space;\end{matrix}&space;\right]&space;=&space;K_{c}\left[&space;\begin{matrix}&space;R_{l}^{c}&space;&space;&&space;t_{l}^{c}&space;&space;&space;\end{matrix}&space;\right]\left[\begin{matrix}&space;x_l&space;&space;\\&space;y_l&space;\\&space;z_l&space;\\&space;1&space;&space;\end{matrix}\right]&space;\quad&space;\quad&space;D(u,v)&space;:=&space;\min\{z_c,D(u,v)\}" title="z_{c} \left[ \begin{matrix} u \\ v \\ 1 \end{matrix} \right] = K_{c}\left[ \begin{matrix} R_{l}^{c} & t_{l}^{c} \end{matrix} \right]\left[\begin{matrix} x_l \\ y_l \\ z_l \\ 1 \end{matrix}\right] \quad \quad D(u,v) := \min\{z_c,D(u,v)\}" />

#### 信息融合定位

基于以下公式，对装甲板框所确定的深度图ROI取均值，作为框中心点的相机坐标系z坐标值，并转换到世界坐标系
<img src="https://latex.codecogs.com/svg.image?\hat&space;z_{c}&space;=&space;\frac{1}{N_{ROI}}\sum_{ROI}D(u,v)&space;\quad&space;\quad&space;&space;\left[\begin{matrix}&space;\hat&space;x&space;&space;\\&space;\hat&space;y&space;\\&space;\hat&space;z&space;&space;\end{matrix}\right]&space;=&space;\left[\begin{matrix}&space;R_c^w&space;&&space;t_c^w&space;&space;\end{matrix}\right]&space;\left[\begin{matrix}{\hat&space;z_c&space;\cdot&space;K_c^{-1}&space;\left[\begin{matrix}&space;u_{center}&space;&space;\\&space;v_{center}&space;\\&space;1&space;\end{matrix}\right]}&space;\\&space;1\end{matrix}\right]" title="\hat z_{c} = \frac{1}{N_{ROI}}\sum_{ROI}D(u,v) \quad \quad \left[\begin{matrix} \hat x \\ \hat y \\ \hat z \end{matrix}\right] = \left[\begin{matrix} R_c^w & t_c^w \end{matrix}\right] \left[\begin{matrix}{\hat z_c \cdot K_c^{-1} \left[\begin{matrix} u_{center} \\ v_{center} \\ 1 \end{matrix}\right]} \\ 1\end{matrix}\right]" />

### 八、程序架构

参考SLAM的架构设计，作为一个机器人项目，我们设计该程序的架构思路如下

<img src="doc_imgs\structure.png" alt="image-20210827150952015" style="zoom:67%;" />

### 九、Road Map

雷达站到战术AI的蜕变之路，都是想法，具体实现还需要调研与探索

<img src="doc_imgs\roadmap.png" alt="image-20210827152216902" style="zoom:67%;" />