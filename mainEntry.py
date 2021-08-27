'''
自定义UI类
使用Qt设计的自定义UI
'''
import sys
from datetime import datetime
import os
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from UART import UART_passer
from radar_class.config import enemy,MAP_PATH,INIT_FRAME_PATH,map_size,VIDEO_SAVE_DIR
from radar_class.camera import read_yaml

from Demo_v4 import Ui_MainWindow

class Mywindow(QtWidgets.QMainWindow, Ui_MainWindow):  # 这个地方要注意Ui_MainWindow
    def __init__(self):
        super(Mywindow, self).__init__()
        self.setupUi(self)
        self.view_change = False # 视角切换控制符
        frame = cv2.imread(INIT_FRAME_PATH)
        frame_m = cv2.imread(MAP_PATH)
        # 小地图翻转
        if enemy:
            frame_m = cv2.rotate(frame_m, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            frame_m = cv2.rotate(frame_m,cv2.ROTATE_90_CLOCKWISE)
        frame_m = cv2.resize(frame_m,map_size)
        self.set_image(frame,"main_demo")
        self.set_image(frame_m,"map")
        del frame,frame_m
        ##############################
        self.feedback_message_box = [] # feedback信息列表

        self.record_object = None # 视频录制对象列表，先用None填充
        # 录制保存位置
        self.save_address = VIDEO_SAVE_DIR
        try:
            if not os.path.exists(self.save_address):
                os.mkdir(self.save_address)
        except: # 当出现磁盘未挂载等情况，导致文件夹都无法创建
            print("[ERROR] The video save dir even doesn't exist on this computer!")

        self.save_title = '' # 当场次录制文件夹名

        # 反馈信息栏，显示初始化
        self.set_text("feedback", "intializing...")
        # 雷达和位姿估计状态反馈栏，初始化为全False
        self.set_text("state", '<br \>'.join(["Radar1: "+ "<font color='#FF0000'><b>False</b></font>",
                                              "Radar2: " + "<font color='#FF0000'><b>False</b></font>",
                                              "Pose1: " + "<font color='#FF0000'><b>False</b></font>",
                                              "Pose2: " + "<font color='#FF0000'><b>False</b></font>"]
                                             ))
        self.record_state = False  # 0:开始 1:停止
        self.btn1.setText("开始录制")
        self.btn2.setText("切换视角 OFF")
        self.btn3.setText("位姿估计")
        self.btn4.setText("open base")

    def btn1_on_clicked(self):
        """
        video recording
        """
        if self.record_state:
            # 结束录制
            self.btn1.setText("开始录制")

            video1,video2 = self.record_object
            video1.release()
            video2.release()

            self.record_object = None

            save_address = os.path.join(self.save_address,self.save_title)

            self.set_text("feedback", "录制已保存于{0}".format(save_address))

            self.record_state = False
        else:
            # 开始录制
            if not os.path.exists(self.save_address):
                print("[ERROR] path not existing")
                return
            self.btn1.setText("停止录制")
            title = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

            self.save_title = title
            os.mkdir(os.path.join(self.save_address,title))
            camera_0_size = read_yaml(0)[4]
            camera_1_size = read_yaml(1)[4]
            self.record_object = [cv2.VideoWriter(os.path.join(self.save_address,title,"1.mp4"),
                                                  cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, camera_0_size),
                                  cv2.VideoWriter(os.path.join(self.save_address,title,"2.mp4"),
                                                  cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, camera_1_size)
                                  ]
            self.record_state = True
            self.set_text("feedback", "录制已开始于{0}".format(title))

    def btn2_on_clicked(self):
        '''
        切换视角
        '''
        if self.view_change:
            self.view_change = False
            self.btn2.setText("切换视角 OFF")
            self.set_text("feedback","view_change off")
        else:
            self.view_change = True
            self.btn2.setText("切换视角 ON")
            self.set_text("feedback","view_change on")

    def btn3_on_clicked(self):
        '''
        位姿估计

        同云台手操作，将裁判系统类的位姿估计置位符设置为真，供主程序处理
        '''
        UART_passer.getposition = True

    def btn4_on_clicked(self):
        '''
        多进程相机视野控制

        同云台手操作，改变裁判系统类的open_base条件判断符
        '''
        if UART_passer.open_base:
            self.btn4.setText("open base")
        else:
            self.btn4.setText("close base")
        UART_passer.open_base = not UART_passer.open_base


    def set_image(self, frame, position=""):
        """
        Image Show Function

        :param frame: the image to show
        :param position: where to show
        :return: a flag to indicate whether the showing process have succeeded or not
        """
        if not position in ["main_demo", "map","message_box"]:
            print("[ERROR] The position isn't a member of this UIwindow")
            return False

        # get the size of the corresponding window
        if position == "main_demo":
            width = self.main_demo.width()
            height = self.main_demo.height()
        elif position == "map":
            width = self.map.width()
            height = self.map.height()
        elif position == "message_box":
            width = self.message_box.width()
            height = self.message_box.height()

        if frame.ndim == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            return False

        # allocate the space of QPixmap
        temp_image = QImage(rgb,rgb.shape[1],rgb.shape[0], QImage.Format_RGB888)

        temp_pixmap = QPixmap(temp_image).scaled(width,height)

        # set the image to the QPixmap location to show the image on the UI
        if position == "main_demo":
            self.main_demo.setPixmap(temp_pixmap)
            self.main_demo.setScaledContents(True)
        elif position == "map":
            self.map.setPixmap(temp_pixmap)
            self.map.setScaledContents(True)
        elif position == "message_box":
            self.message_box.setPixmap(temp_pixmap)
            self.message_box.setScaledContents(True)
        return True

    def set_text(self, position: str, message=""):
        """
        to set text in the QtLabel

        :param position: must be one of the followings: "feedback", "message_box", "state"
        :param message: For feedback, a string you want to show in the next line;
        For the others, a string to show on that position , which will replace the origin one.
        :return:
        a flag to indicate whether the showing process have succeeded or not
        """
        if position not in ["feedback", "message_box","state"]:
            print("[ERROR] The position isn't a member of this UIwindow")
            return False
        if position == "feedback":
            if len(self.feedback_message_box) >= 12: # the feedback could contain at most 12 messages lines.
                self.feedback_message_box.pop(0)
            self.feedback_message_box.append(message)
            # Using "<br \>" to combine the contents of the message list to a single string message
            message = "<br \>".join(self.feedback_message_box) # css format to replace \n
            self.feedback.setText(message)
            return True
        if position == "state":
            self.state.setText(message)
            return True
        if position == "message_box":
            self.message_box.setText(message)
            return True


if __name__ == "__main__":
    # demo of the window class
    app = QtWidgets.QApplication(sys.argv)
    myshow = Mywindow()

    myshow.set_text("feedback",'<br \>'.join(['',"<font color='#FF0000'><b>base detect enermy</b></font>","<font color='#FF0000'><b>base detect enermy</b></font>",
                                  f"哨兵:<font color='#FF0000'><b>{99:d}</b></font>"]))

    myshow.show()  # 显示

    frame_m = cv2.imread(MAP_PATH)
    while True:
        # 需要循环show一个opencv窗口才能保证QtUI的更新
        cv2.imshow("out",frame_m)
        key = cv2.waitKey(80)
        if key == ord('q')&0xff:
            break
    sys.exit(app.exec_())
