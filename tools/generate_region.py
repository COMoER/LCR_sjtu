'''
This a useful tool script to help you use picking points or selecting ROI to choose your favourite region
and to reverse the red region you have picked to the blue side or the to resize the region to your favourite real_size,
although it's not a part of the main program.
'''
import cv2
import numpy as np
from radar_class.config import region

MAP_DIR = "../../yolov5_4.0_radar_new/demo_resource/map2.jpg"

def change_region(old_r,new_size,origin_size):
    '''
    将某种真实赛场比例下的区域转化为新规定的赛场比例（new_size)输出

    本函数输出高度均设置为0
    '''
    for r in old_r.keys():
        _,f_type,_,_,_ = r.split('_')
        if f_type == 'l':
            thre =  old_r[r][4]
            rect = np.float32(old_r[r][:4]).reshape(-1,2)
            for i in range(2):
                rect[:,i] = rect[:,i]/origin_size[i]*new_size[i]
            x1,y1,x2,y2 = rect.reshape(-1)
            print("\'{0}\':[{1:.3f},{2:.3f},{3:.3f},{4:.3f},{5:.3f}],".format(r,x1,y1,x2,y2,thre ))
        if f_type == 'r':
            rect = np.float32(old_r[r][:4]).reshape(-1,2)

            for i in range(2):
                rect[:,i] = rect[:,i]/origin_size[i]*new_size[i]
            x1,y1,x2,y2 = rect.reshape(-1)
            print("\'{0}\':[{1:.3f},{2:.3f},{3:.3f},{4:.3f},0.,0.],".format(r,x1,y1,x2,y2))
        if f_type == 'fp':
            rect = np.float32(old_r[r][:8]).reshape(-1,2)
            for i in range(2):
                rect[:,i] = rect[:,i]/origin_size[i]*new_size[i]
            rect = rect.reshape(-1)
            print("\'{0}\':[{1:.3f},{2:.3f},{3:.3f},{4:.3f},{5:.3f},{6:.3f},{7:.3f},{8:.3f},0.,0.],".format(r,
                                                                                                           *rect))


def region_plot(region:dict):
    '''
    在小地图上画区域

    一个区域一个区域显示
    '''
    frame_m = cv2.imread(MAP_DIR)
    size = frame_m.shape[1], frame_m.shape[0]
    cv2.namedWindow("out",cv2.WINDOW_NORMAL)
    map_c = frame_m.copy()
    for r in region.keys():
        ftype,alarm_type,_,_,l_type = r.split('_')
        if ftype == 'm' or ftype == 'a':
            if alarm_type == 'fp':
                f = lambda x:(int(x[0] * size[0] // real_size[0]),
                                   int((15. - x[1]) * size[1] // real_size[1]))
                rect = np.array(region[r][:8]).reshape(4,2)  # rect(x0,y0,x1,y1)
                print("now plot region is {0}".format(r))
                for i in range(4):
                    cv2.line(map_c,f(rect[i]),f(rect[(i+1)%4]),(0,255,0),2)
                    cv2.imshow("out",map_c)
                cv2.waitKey(0) # 调整waitkey的位置，也可以一条线一条线显示
    cv2.destroyWindow("out")
def reverse(region,real_size):
    '''
    翻转，根据赛场中心对称原则将红方的区域位置转化为蓝方的
    高度均设为0
    '''
    for r in region.keys():
        ftype,shape_type,team,loc,htype = r.split('_')
        if team == 'red':
            if shape_type == 'r' or shape_type == 'l':
                ps  = np.array(region[r][:4]).reshape(-1,2)
            else:
                ps = np.array(region[r][:8]).reshape(-1, 2)
            ps[:,0] = real_size[0]-ps[:,0]
            ps[:,1] = real_size[1]-ps[:,1]

            name = "{0}_{1}_{2}_{3}_{4}".format(ftype,shape_type,'blue',loc,htype)
            if shape_type == 'l':
                x1, y1, x2, y2 = ps.reshape(-1).tolist()
                thres = region[r][4]
                print("\'{0}\':[{1:.3f},{2:.3f},{3:.3f},{4:.3f},{5:.3f}],".format(name, x2, y2, x1, y1,thres))
            if shape_type == 'r':
                x1, y1, x2, y2 = ps.reshape(-1).tolist()
                print("\'{0}\':[{1:.3f},{2:.3f},{3:.3f},{4:.3f},0.,0.],".format(name, x2, y2, x1, y1))
            if shape_type == 'fp':
                x1,y1,x2,y2,x3,y3,x4,y4 = ps.reshape(-1)
                print("\'{0}\':[{1:.3f},{2:.3f},{3:.3f},{4:.3f},{5:.3f},{6:.3f},{7:.3f},{8:.3f},0.,0.],".format(name,x3,y3,x4,y4,x1,y1,x2,y2))

def generate_rect(real_size):
    '''
    generate rect region

    通过手动选ROI框来生成矩形区域,输入格式为"{team}_{location_name}"
    '''
    frame_m = cv2.imread(MAP_DIR)
    map_size = frame_m.shape[1],frame_m.shape[0]
    cv2.namedWindow("out",cv2.WINDOW_NORMAL)
    fx = lambda x:x/map_size[0]*real_size[0]
    fy = lambda y:real_size[1]-y/map_size[1]*real_size[1]
    while True:
        name = input("input region name\n")
        cv2.imshow("out", frame_m)
        rect = cv2.selectROI("out",frame_m,False)
        print("*"*20)
        im = frame_m.copy()
        cv2.rectangle(im,rect[:2],(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),2)
        x1 = fx(rect[0])
        y1 = fy(rect[1])
        x2 = fx(rect[0]+rect[2])
        y2 = fy(rect[1]+rect[3])
        print("\'m_r_{0}_a\':[{1:.3f},{2:.3f},{3:.3f},{4:.3f},0.,0.],".format(name,x1,y1,x2,y2))
        cv2.imshow("out",im)

        key = cv2.waitKey(0)
        if key == ord('q')&0xFF:
            break
    cv2.destroyWindow("out")
def __callback_1(event,x,y,flags,param):

    if event == cv2.EVENT_LBUTTONDOWN and not param['pick_flag']:
        param['pick_flag'] = True
        param['p'] = [x,y]
        cv2.circle(param['img'],(x,y),2,(0,255,0),-1)
        print(f"pick ({x:d},{y:d})")

def fp_generate(real_size):
    '''
    基于类似于四点位姿估计标定的方式，进行四点区域选取，输入格式为"{team}_{location_name}"
    '''
    frame_m = cv2.imread(MAP_DIR)
    name = input("input region name\n")
    map_size = frame_m.shape[1],frame_m.shape[0]
    fx = lambda x:x/map_size[0]*real_size[0]
    fy = lambda y:real_size[1]-y/map_size[1]*real_size[1]
    cv2.namedWindow("pick", cv2.WINDOW_NORMAL)
    info = {"pick_flag":False,'img':frame_m.copy(),'p_img':frame_m.copy()}
    cv2.setMouseCallback("pick", __callback_1,info)
    pick_point = []
    while True:
        cv2.imshow("pick", info['img'])
        if info['pick_flag']:
            if cv2.waitKey(0) == ord('c') & 0xFF:
                pick_point.append(info['p'])
                print(f"You have pick {len(pick_point):d} point.")
                info['p_img'] = info['img'].copy()
            else:
                info['img'] = info['p_img'].copy()
            info['pick_flag'] = False
            if len(pick_point) == 4:
                break
        else:
            cv2.waitKey(1)
    for p in pick_point:
        p[0] = fx(p[0])
        p[1] = fy(p[1])
    pick_point = np.array(pick_point).reshape(-1)
    print("\'m_fp_{0}_a\':[{1:.3f},{2:.3f},{3:.3f},{4:.3f},{5:.3f},{6:.3f},{7:.3f},{8:.3f},0.,0.]".format(name,*pick_point))
    print("*"*20)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    real_size = (28.,15.)
    # 使用例
    print("#"*20)
    print("change_region output:")
    change_region(region,(12.,6.),real_size) # 基于比赛region生成家里测试region
    print("#"*20)
    print("reverse output:")
    reverse(region,real_size)
    print("#"*20)
    region_plot(region)
    print("###### generate rect region ######")
    generate_rect(real_size)
    print("###### generate fp region ######")
    fp_generate(real_size)