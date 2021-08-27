'''
default network class
给神经网络类的接口格式定义，神经网络具体需要自行添加
'''
import pickle as pkl


class Predictor(object):
    def __init__(self,weights = ""):
        '''
        对于示例类，不会提供神经网络预测功能，但对于我们提供的demo，可以加载pkl来获得实际的预测结果

        :param weights:pkl文件的存放地址
        '''
        self._weights = weights
        with open(self._weights,'rb') as net:
            self._predicted_data = pkl.load(net)

    def infer(self,imgs,id):
        '''
        这个函数用来预测
        :param imgs:list of input images

        :return:
        img_preds: 车辆预测框，a list of the prediction to each image 各元素格式为(predicted_class,conf_score,bounding box(format:x0,y0,x1,y1))

        car_locations: 对于每张图片装甲板预测框（车辆定位） np.ndarray 和对应的车辆预测框(与装甲板预测框的车辆预测框序号对应）的列表
        上述两个成员具体定义为：
        （1）装甲板预测框格式,(N,装甲板四点+装甲板网络置信度+装甲板类型+其对应的车辆预测框序号（即其为哪个车辆预测框ROI区域预测生成的）+四点的bounding box)
        其他敌方提到该格式，会写为（N,fp+conf+cls+img_no+bbox)

        （2）车辆预测框格式 np.ndarray (N,x0+y0+x1+y1)
        '''
        img_preds,car_locations = self._predicted_data[id]
        return img_preds,car_locations
