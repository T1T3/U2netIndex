from typing import Any
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import math
from skimage import morphology


class LineDirection():
    def __init__(self, Image,Center,Start_Pos,End_Pos):
        self.image = Image    # gray image
        self.center = (Center[0],Center[1])
        self.start_pos = (Start_Pos[0],Start_Pos[1])
        self.end_pos = (End_Pos[0],End_Pos[1])
        self.D1_point = None
        self.D2_point = None
        self.D3_point = None
        self.D3_point_LR = None

        angle_start = math.atan2(self.start_pos[1] - self.center[1], self.start_pos[0] - self.center[0])
        angle_end = math.atan2(self.end_pos[1] - self.center[1], self.end_pos[0] - self.center[0])
        print(angle_start, angle_end)

        # 角度
        angle_start_degrees = (math.degrees(angle_start)+ 360) % 360
        angle_end_degrees = (math.degrees(angle_end)+ 360) % 360
        
        print(angle_start_degrees, angle_end_degrees)
        

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        D1image=self.image.copy()
        D2image=self.image.copy()
        D3image=self.image.copy()
        self.D1_point=self.Direction_1(D1image)
        self.D2_point=self.Direction_2(D2image,self.D1_point[1]) # direction for D2
        self.D3_point, self.D3_point_LR =self.Direction_3(D3image)
        converted_data_LR = ([int(x) for x in self.D3_point_LR[0]], [int(x) for x in self.D3_point_LR[1]])
        self.D3_point_LR=[converted_data_LR]
        print("-----LR--------",self.D3_point_LR )
        # take 3 angle
        R1,R2,R3,D1_point, D2_point, D3_point=self.draw_lines(self.image, self.D1_point, self.D2_point, self.D3_point, self.D3_point_LR)
        
        return R1,R2,R3,D1_point, D2_point, D3_point
        

    def Direction_1(self, image):
        image_center = self.center
        max_distance = 0
        max_points = None
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        epsilon = 0.01 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        image_with_hull = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.polylines(image_with_hull, [hull], True, (0, 255, 0), 2)
        cv2.polylines(image_with_hull, [approx], True, (0, 0, 255), 2)

        # Draw the image center
        cv2.circle(image_with_hull, image_center, 5, (255, 0, 0), -1)

        # # Find the farthest point from the center on the convex hull
        # farthest_dist = 0
        # farthest_point = None
        # for point in approx[:, 0, :]:
        #     dist = np.linalg.norm(point - image_center)
        #     if dist > farthest_dist:
        #         farthest_dist = dist
        #         farthest_point = tuple(point)

        points=approx[:, 0, :]

        # 计算每对像素点之间的距离
        distances = squareform(pdist(points))
        # 找到最长距离的1/4作为阈值
        threshold = np.max(distances) / 4

        kmeans = KMeans(n_clusters=math.ceil(len(points) / 2), random_state=0)
        # 拟合模型并预测
        kmeans.fit(points)
        labels = kmeans.labels_

        # 分组
        group=[]
        centers=kmeans.cluster_centers_
        center_point=[]
        group_lens=[]
        for i in set(labels):
            group.append(points[labels == i])
            center_point.append((centers[i].astype(int)))
            group_lens.append(len(points[labels == i]))

        # 长度，闭包,2组以上时候
        def calculate_length(vertices):
            '''
            input:[[ x1 , y1 ] , [ x2 , y2 ]]
            output:len
            '''
            total_length = 0
            num_vertices = len(vertices)
            if num_vertices == 2:
                return np.sqrt((vertices[1][0] - vertices[0][0])**2 + (vertices[1][1] - vertices[0][1])**2)
            elif num_vertices > 2:
                for i in range(num_vertices):
                    j = (i + 1) % num_vertices  
                    total_length += np.sqrt((vertices[j][0] - vertices[i][0])**2 + (vertices[j][1] - vertices[i][1])**2)
            return total_length

        sorted_cp_with_index = sorted(enumerate(center_point), key=lambda x: max(np.linalg.norm(np.array(x[1]) - np.array(y)) for y in center_point), reverse=True)[:2]

        #point检测
        print("shede",sorted_cp_with_index)

        # 1:1 point type
        if group_lens[sorted_cp_with_index[0][0]]==1 and group_lens[sorted_cp_with_index[1][0]]==1:
            # Find the farthest point from the center on the convex hull
            farthest_dist = 0
            farthest_point = None
            for point in approx[:, 0, :]:
                dist = np.linalg.norm(point - image_center)
                if dist > farthest_dist:
                    farthest_dist = dist
                    farthest_point = tuple(point)
            print("res:",farthest_point)
            return image_center,farthest_point
        
        # 1:n point type
        elif group_lens[sorted_cp_with_index[0][0]]==1:
            print("res:",sorted_cp_with_index[0][1])
            return image_center,(sorted_cp_with_index[0][1])
        
        # n:1 point type
        elif group_lens[sorted_cp_with_index[1][0]]==1:
            print("res:",sorted_cp_with_index[1][1])
            return image_center,(sorted_cp_with_index[1][1])
        
        # n:n point type
        else:
            if calculate_length(group[sorted_cp_with_index[0][0]])<calculate_length(group[sorted_cp_with_index[1][0]]):
                print("res:",sorted_cp_with_index[0][1])
                return image_center,(sorted_cp_with_index[0][1])
            else:
                print("res:",sorted_cp_with_index[1][1])
                return image_center,(sorted_cp_with_index[1][1])

        
        # for i in range(len(approx[:, 0, :])):
        #     for j in range(i + 1, len(approx[:, 0, :])):
        #         # 计算点对之间的距离
        #         distance = np.linalg.norm(points[i] - points[j])
        #         # 如果当前距离大于最大距离，则更新最大距离和最大点对
        #         if distance > max_distance:
        #             max_distance = distance
        #             max_points = (points[i], points[j])

        # point1, point2 = max_points
        # # 计算最远的两对点的直线方程 y = mx + b
        # m1 = (point2[1] - point1[1]) / (point2[0] - point1[0])
        # b1 = point1[1] - m1 * point1[0]
        # # 另一条线的斜率和截距
        # m2 = -1 / m1  # 垂直于第一条线的斜率的负倒数
        # b2 = point1[1] - m2 * point1[0]

        # x_intersect = (b2 - b1) / (m1 - m2)
        # # 代入其中一个方程得到 y 坐标
        # y_intersect = m1 * x_intersect + b1

        # print("------test--------")
        # print(approx[:, 0, :])
        # print((x_intersect, y_intersect))
        # print("最远的两对点：", point1, point2)
        # print("------------------")

        # return image_center,(int(x_intersect), int(y_intersect))


    def Direction_2(self, image,direction):
        image_center = self.center
        # edges = cv2.Canny(image, 50, 150, apertureSize=3)
        _,binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)  # 二值化处理
        # cv2.imwrite("binary.png", binary)   # 保存二值化图片

        # 骨架提取 细化
        binary[binary==255] = 1
        skeleton0 = morphology.skeletonize(binary)   
        skeleton = skeleton0.astype(np.uint8)*255

        # plt.imshow(skeleton)
        # plt.show()

        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.arcLength(x, False), reverse=True)

        largest_contours = contours[0]

        # 初始化距离最近的直线
        closest_line = None
        # min_distance = float('inf')

        # 对每个轮廓进行线性拟合并绘制直线
        #for contour in largest_contours:
            # 过滤10像素以下的轮廓
            # print("cv2.contourArea(contour)",cv2.contourArea(contour))
            # if cv2.contourArea(contour) < 10 :
            #     continue
        
        # 线性拟合 [[cos a],[sin a],[point_x],[point_y]] Y轴正半轴
        # 在一个image(640,480)大小的图像中取得一个拟合的直线
        [vx, vy, x, y] = cv2.fitLine(largest_contours, cv2.DIST_L12, 0, 0.1, 0.1)

        # 计算直线的斜率和截距
        slope = vy / vx if vx != 0 else np.inf
        intercept = y - slope * x if vx != 0 else np.inf
        # 计算交点
        # 如果直线平行于y轴
        if np.isinf(slope):
            # 直线与左右两边界相交
            pt1 = (0, int(intercept))
            pt2 = (image.shape[1] - 1, int(image.shape[1] * slope + intercept))
        # 如果直线平行于x轴
        elif slope == 0:
            # 直线与上下两边界相交
            pt1 = (int(intercept), 0)
            pt2 = (int(image.shape[0] * slope + intercept), image.shape[0] - 1)
        else:
            # 计算直线与上下左右四个边界的交点
            pt1 = (int((0 - intercept) / slope), 0)
            pt2 = (int((image.shape[0] - 1 - intercept) / slope), image.shape[0] - 1)
        # 修正交点坐标，确保在图像范围内
        _,pt1,pt2=cv2.clipLine((0, 0, image.shape[1], image.shape[0]), pt1, pt2)
        print("pt",pt1,pt2)
        # test绘制交点
        #cv2.circle(image, pt1, 5, (255, 0, 0), -1)
        #cv2.circle(image, pt2, 5, (255, 0, 0), -1)
        cv2.line(image, pt2, pt1, (0, 255, 0), 2)

        # 计算距离
        dist_pt1 = np.linalg.norm(np.array(pt1) - np.array(direction))
        dist_pt2 = np.linalg.norm(np.array(pt2) - np.array(direction))

        # 选择最近的点
        if dist_pt1 < dist_pt2:
            nearest_point = pt1
        else:
            nearest_point = pt2

        # lefty = int((-x * vy / vx) + y)
        # righty = int(((image.shape[1] - x) * vy / vx) + y)


        # cv2.line(image, (image.shape[1] - 1, righty), (0, lefty), (0, 255, 0), 2)
        # closest_line = ((0, lefty),(image.shape[1] - 1, righty), )

        # # 计算直线与给定点的距离
        distance = np.abs((vy * (direction[0] - x) - vx * (direction[1] - y)) / np.sqrt(vx**2 + vy**2))
        print("Distance wiht D1 point:",distance)

        #     # 绘制直线

        # # 如果当前直线更接近中心点，更新最小距离和最近的直线
        # if distance < min_distance:
        #     min_distance = distance
        #     cv2.line(image, (image.shape[1] - 1, righty), (0, lefty), (255, 0, 0), 1)
            
        #     closest_line = ((0, lefty),(image.shape[1] - 1, righty), )
            
        #     # return closest_line
                
        plt.imshow(image)
        plt.show()

        return (image_center,nearest_point)

    def Direction_3(self, image):
        # plt.imshow(image)
        # plt.show()

        edges = cv2.Canny(image, 50, 150, apertureSize=3,L2gradient = True)
        # plt.imshow(edges)
        # plt.show()
        # HoughLinesP to find linesd
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

        # Sort the lines based on length
        sorted_lines = sorted(lines, key=self.get_line_length, reverse=True)
        lines4 = sorted_lines[:3]

        min_distance=float('inf')
        # 变换
        line_strings = [LineString([(point[0][0], point[0][1]), (point[0][2], point[0][3])]) for point in lines4]

        for i in range(len(line_strings)-1):
            for j in range(i+1, len(line_strings)):
                distance = line_strings[i].distance(line_strings[j])
                if distance < min_distance and distance > 1:
                    min_distance = distance
                    longest_two_lines = (list(line_strings[i].coords), list(line_strings[j].coords))

                # print(f"Distance between Line {i+1} and Line {j+1}: {distance}")
        # longest_two_lines = sorted_lines[1:3] # [:2]
        # print(longest_two_lines)
        # Calculate the intersection point of the longest two lines
        intersection_point = self.line_intersection(longest_two_lines[0], longest_two_lines[1])

        if len(longest_two_lines[0]) == 2:
            return (self.center,intersection_point), ((longest_two_lines[0][0][0],longest_two_lines[0][0][1],longest_two_lines[0][1][0],longest_two_lines[0][1][1]),
                                                      (longest_two_lines[1][0][0],longest_two_lines[1][0][1],longest_two_lines[1][1][0],longest_two_lines[1][1][1]))
        else:
            return (self.center,intersection_point), longest_two_lines


    def draw_lines(self,
                   image,
                   D1_point,
                   D2_point,
                   D3_point,
                   D3_point_LR):
        '''
        draw the lines in the image
        '''
        image_color=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # print("D1:",D1_point[0], D1_point[1])
        # print("D2:",D2_point[0], D2_point[1])
        # print("D3:",D3_point[0], D3_point[1])
        # print("D3lr:",D3_point_LR)

        cv2.line(image_color, D1_point[0], D1_point[1], (0, 0, 255), 1)
        cv2.line(image_color, D2_point[0], D2_point[1], (0, 255, 0), 1)
        cv2.line(image_color, D3_point[0], D3_point[1], (255, 0, 0), 1)
        for line in D3_point_LR:
            for x1, y1, x2, y2 in line:
                cv2.line(image_color, (x1, y1), (x2, y2), (155, 155, 0), 1)

        # 纠正正方向，计算每对点之间的距离
        distances = [
            (D1_point[1], D2_point[0], math.dist(D1_point[1], D2_point[0])),
            (D1_point[1], D2_point[1], math.dist(D1_point[1], D2_point[1])),
            (D3_point[1], D2_point[0], math.dist(D3_point[1], D2_point[0])),
            (D3_point[1], D2_point[1], math.dist(D3_point[1], D2_point[1])),]
        distances.sort(key=lambda x: x[2])
        closest_points = distances[0][:2]
        distance = distances[0][2]

        # D1_angle = self.get_angle( D1_point[0][0], D1_point[0][1], D1_point[1][0], D1_point[1][1])
        # D2_angle = self.get_angle( D2_point[0][0], D2_point[0][1], D2_point[1][0], D2_point[1][1])
        # D3_angle = self.get_angle( D3_point[0][0], D3_point[0][1], D3_point[1][0], D3_point[1][1])

        D1_SA_angle = self.clockwise_angle( (self.center,self.start_pos),(D1_point[0], D1_point[1]) )
        D2_SA_angle = self.clockwise_angle( (self.center,self.start_pos),(self.center, closest_points[1]))
        # D2_SA_angleSP = self.clockwise_angle((self.center,self.start_pos), (D2_point[0], D2_point[1]))
        D3_SA_angle = self.clockwise_angle( (self.center,self.start_pos),(D3_point[0], D3_point[1]))


        # plt.figure(figsize=(10, 10))
        # plt.imshow(image_color)
        # plt.title(f"Arrow Direction:[Blue]{D1_SA_angle:.2f}, [Green]{D2_SA_angle:.2f}/{D2_SA_angleSP:.2f}, [Red]{D3_SA_angle:.2f}")
        # plt.show()


        return D1_SA_angle, D2_SA_angle , D3_SA_angle, D1_point, D2_point, D3_point
    
    def get_line_length(self,line):
        x1, y1, x2, y2 = line[0]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    # Function to calculate the intersection point of two lines
    def line_intersection(self,line1, line2):
        if len(line1)==2:
            x1, y1 = line1[0]
            x2, y2 = line1[1]
            x3, y3 = line2[0]
            x4, y4 = line2[1]
        elif len(line1)==4:
            x1, y1, x2, y2 = line1[0]
            x3, y3, x4, y4 = line2[0]

        px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / \
            ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10)
        py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / \
            ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10)
        return (int(px), int(py))

    # Calculate the angle of the line with respect to the 12 o'clock position
    def get_angle(self, Cx, Cy, Px, Py):
        '''
        Calculate the angle of the line with respect to the 12 o'clock position

        '''

        dx = Px - Cx
        dy = Py - Cy 
        angle = (np.arctan2(dy, dx) * 180.0 / np.pi) % 360  # Convert radian to degree and ensure positive angle
        # 90-angle
        angle_from_startpos = (angle+90) % 360
        angle_from_12 = (angle+90) % 360  # Adjust angle to the 12 o'clock reference
        print("angle_from_12:",angle_from_12)
        return angle_from_12
    

    def clockwise_angle(self, line1, line2):
        c1,p1 = line1 # 中心点坐标 c1 == c2 相等
        c2,p2 = line2 # 中心点坐标 c1 == c2 相等
        c1 = np.array(c1)
        v1 = [p1[0] - c1[0], p1[1] - c1[1]]
        v2 = [p2[0] - c1[0], p2[1] - c2[1]]
        x1, y1 = v1
        x2, y2 = v2
        dot = x1 * x2 + y1 * y2
        det = x1 * y2 - y1 * x2
        theta = np.arctan2(det, dot)
        theta = theta if theta > 0 else 2 * np.pi + theta
        return theta * (180 / np.pi)    
    
    def Disttances(a, b):
        #返回两点间距离
        x1, y1 = a
        x2, y2 = b
        Disttances = int(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        return Disttances


if __name__ == '__main__':
    # test code
    image_center = (338, 232)
    # image_path = r'D:\git\Deamnet\0221tests_imgD2\20240123_133704point_mask.jpg'

    image_path = r'tests_out0x00000002_20240120030318077_imgpoint_mask.jpg'
    output_path = r'tests_out2'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    lineDirection1 = LineDirection( Image=image,Center=image_center,Start_Pos=(195, 395),End_Pos=(547,393))
    lineDirection1()
    