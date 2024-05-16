import math
import os
import sys


import cv2
import matplotlib.pyplot as plt
import numpy
import torch

from models.net import U2NET


class Tester(object):

    def __init__(self):
        self.net = U2NET(3, 2)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load('weight/net0515_449_37350.pt', map_location='cpu'))
        self.net.eval().to(self.device)

        '''以下为超参数，需要根据不同表盘类型设定'''
        self.line_width = 1000  # 表盘展开为直线的长度
        self.line_height = 150  # 表盘展开为直线的宽度
        self.circle_radius = 512 # 200  # 预设圆盘直径
        self.circle_center = [306, 237]# [208, 208]  # 圆盘指针的旋转中心
        self.pi = 3.1415926535898


    @torch.no_grad()
    def __call__(self, image,image_name='', output_path='./output/'):
        image_name = image_name
        output_path = output_path
        
        try:
            image_tensor = self.to_tensor(image.copy()).to(self.device)
        except :
            print("!---ERROR :",image_name)
            return None
        d0, d1, d2, d3, d4, d5, d6 = self.net(image_tensor)
        mask = d0.squeeze(0).cpu().numpy()
        point_mask = self.binary_image(mask[0])
        dail_mask = self.binary_image(mask[1])
        relative_value = self.get_relative_value(point_mask, dail_mask,output_path)

        cv2.putText(image, f"relative_value: {relative_value['ratio']:.6f} degrees", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        
        
        index_mask = point_mask.copy()

        if index_mask is not None :
            img_converted = numpy.where(index_mask != 0, 255, index_mask)
            img_converted = index_mask * 255
            cv2.imwrite(output_path + image_name + 'point_mask.jpg' , img_converted)


        #cv2.imwrite(output_path + image_name + 'dail_mask.jpg' , dail_mask)
        condition = point_mask ==1
        image[condition] = (0,0,255)
        condition = dail_mask == 1
        image[condition] = (0, 255, 0)
        # cv2.imshow('image_mask', image)
        # cv2.waitKey()
        
        # save images
        #cv2.imwrite(output_path + image_name + '_mask.jpg' , image)
        return None
    
    def detectU2net(self, image):
        image_tensor = self.to_tensor(image.copy()).to(self.device)
        d0, d1, d2, d3, d4, d5, d6 = self.net(image_tensor)
        mask = d0.squeeze(0).cpu().detach().numpy()
        point_mask = self.binary_image(mask[0])
        dail_mask = self.binary_image(mask[1])
        # relative_value = self.get_relative_value(point_mask, dail_mask,None)
        
        gray_mask = (point_mask * 255).astype(numpy.uint8)

        condition = point_mask ==1
        image[condition] = (0,0,255)
        condition = dail_mask == 1
        image[condition] = (0, 255, 0)


        return image,gray_mask

    def binary_image(self, image):
        condition = image > 0.5
        image[condition] = 1
        image[~condition] = 0
        image = self.corrosion(image)
        return image

    def get_relative_value(self, image_pointer, image_dail,output_path):
        line_image_pointer = self.create_line_image(image_pointer)
        line_image_dail = self.create_line_image(image_dail)
        data_1d_pointer = self.convert_1d_data(line_image_pointer)
        data_1d_dail = self.convert_1d_data(line_image_dail)
        data_1d_dail = self.mean_filtration(data_1d_dail)

        _, ax = plt.subplots()
        # plt.plot(numpy.arange(0,len(data_1d_pointer)),data_1d_pointer)
        # plt.plot(numpy.arange(0, len(data_1d_dail)), data_1d_dail)
        ax.plot(numpy.arange(0, len(data_1d_pointer)), data_1d_pointer, label='Pointer')
        ax.plot(numpy.arange(0, len(data_1d_dail)), data_1d_dail, label='Dail')

        # plt save
        if output_path==None :
            pass
        else:
            plt.savefig(output_path + image_name.split('.')[0] + '_ratio.jpg')
            plt.close()
        
        # plt.show()

        # SAVE IMAGES
        # cv2.imwrite(output_path + image_name + 'line_image_pointer.jpg' , line_image_pointer)
        # cv2.imwrite(output_path + image_name + 'line_image_dail.jpg' , line_image_dail)
        # cv2.imshow('line_image_pointer', line_image_pointer)
        # cv2.imshow('line_image_dail', line_image_dail)

        '''定位指针相对刻度位置'''
        dail_flag = False
        pointer_flag = False
        one_dail_start = 0
        one_dail_end = 0
        one_pointer_start = 0
        one_pointer_end = 0
        dail_location = []
        pointer_location = 0
        for i in range(self.line_width - 1):
            if data_1d_dail[i] > 0 and data_1d_dail[i + 1] > 0:
                if not dail_flag:
                    one_dail_start = i
                    dail_flag = True
            if dail_flag:
                if data_1d_dail[i] == 0 and data_1d_dail[i + 1] == 0:
                    one_dail_end = i - 1
                    one_dail_location = (one_dail_start + one_dail_end) / 2
                    dail_location.append(one_dail_location)
                    one_dail_start = 0
                    one_dail_end = 0
                    dail_flag = False
            if data_1d_pointer[i] > 0 and data_1d_dail[i + 1] > 0:
                if not pointer_flag:
                    one_pointer_start = i
                    pointer_flag = True
            if pointer_flag:
                if data_1d_pointer[i] == 0 and data_1d_pointer[i + 1] == 0:
                    one_pointer_end = i - 1
                    pointer_location = (one_pointer_start + one_pointer_end) / 2
                    one_pointer_start = 0
                    one_pointer_end = 0
                    pointer_flag = False
        scale_num = len(dail_location)
        num_scale = -1
        ratio = -1
        if scale_num > 0:
            for i in range(scale_num - 1):
                if dail_location[i] <= pointer_location < dail_location[i + 1]:
                    num_scale = i + (pointer_location - dail_location[i]) / (
                            dail_location[i + 1] - dail_location[i] + 1e-5) + 1
            ratio = (pointer_location - dail_location[0]) / (dail_location[-1] - dail_location[0] + 1e-5)
        result = {'scale_num': scale_num, 'num_sacle': num_scale, 'ratio': ratio}
        return result

    def create_line_image(self, image_mask):
        line_image = numpy.zeros((self.line_height, self.line_width), dtype=numpy.uint8)
        for row in range(self.line_height):
            for col in range(self.line_width):
                """计算与-y轴的夹角"""
                theta = ((2 * self.pi) / self.line_width) * (col + 1)
                '''计算当前扫描点位对应于原图的直径'''
                radius = self.circle_radius - row - 1
                '''计算当前扫描点对应于原图的位置'''
                y = int(self.circle_center[0] + radius * math.cos(theta) + 0.5)
                x = int(self.circle_center[1] - radius * math.sin(theta) + 0.5)
                # print(radius,y,x)
                try:
                    line_image[row, col] = image_mask[y, x]
                except IndexError:
                    pass

        # plt.imshow(line_image)
        # plt.show()
        return line_image

    def convert_1d_data(self, line_image):
        """
        将图片转换为1维数组
        :param line_image: 展开的图片
        :return: 一维数组
        """
        data_1d = numpy.zeros((self.line_width), dtype=numpy.int16)
        for col in range(self.line_width):
            for row in range(self.line_height):
                if line_image[row, col] == 1:
                    data_1d[col] += 1
        return data_1d

    def corrosion(self, image):
        """
        腐蚀操作
        :param image:
        :return:
        """
        kernel = numpy.ones((3, 3), numpy.uint8)
        image = cv2.erode(image, kernel)
        return image

    def mean_filtration(self, data_1d_dail):
        """
        均值滤波
        :param data_1d_dail:
        :return:
        """
        mean_data = numpy.mean(data_1d_dail)
        for col in range(self.line_width):
            if data_1d_dail[col] < mean_data:
                data_1d_dail[col] = 0
        return data_1d_dail

    @staticmethod
    def to_tensor(image):
        image = torch.tensor(image).float() / 255
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image

    @staticmethod
    def square_picture(image, image_size):
        """
        任意图片正方形中心化
        :param image: 图片
        :param image_size: 输出图片的尺寸
        :return: 输出图片
        """
        h1, w1, _ = image.shape
        max_len = max(h1, w1)
        fx = image_size / max_len
        fy = image_size / max_len
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        h2, w2, _ = image.shape
        background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
        background[:, :, :] = 127
        s_h = image_size // 2 - h2 // 2
        s_w = image_size // 2 - w2 // 2
        background[s_h:s_h + h2, s_w:s_w + w2] = image
        return background


if __name__ == '__main__':

    root = 'data/images/val'
    OutputPath= 'data/results/'
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)
    tester = Tester()
    for image_name in os.listdir(root):
        path = f'{root}/{image_name}'
        image = cv2.imread(path)
        tester(image,image_name.split('.')[0],output_path=OutputPath)
