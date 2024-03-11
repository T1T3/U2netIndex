import pygame
from pygame.locals import *
import sys
import math
import cv2
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

from LineDirection import LineDirection
from test import Tester



class IndexVeiwer(object):
    def __init__(self,object,image_path):
        self.txtfile_path = 'output.txt'
        if not os.path.exists(self.txtfile_path):
            open(self.txtfile_path, 'w').close()

        self.Tester = object
        # Initialize Pygame
        pygame.init()
        self.width, self.height = 640, 480
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Dystopian Sector Visualizer")

        # Load background image
        self.IMAGE=cv2.imread(image_path)
        self.background_image = pygame.image.load(image_path)
        self.background_image = pygame.transform.scale(self.background_image, (self.width, self.height))

        # Colors
        self.black = (0, 0, 0)
        # self.green = (0, 255, 0, 100)  # Semi-transparent green

        # Variables
        self.center = [0, 0]
        self.clicked = False
        self.drawing = False
        self.start_pos = [0, 0]
        self.radius = 0

        # position for 0 point to end point
        self.PosiationX=0
        self.PosiationY=0

        self.U2netON=False
        # radius for DetectU2net
        self.R1=0
        self.R2=0
        self.R3=0
        # defult range
        self.Range=1.5

        self.tmp =(0,0)

        # use pygame take text on the screen.
        pygame.font.init()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 30)
    
    def __call__(self, image_path):
        self.screen.blit(self.background_image, (0, 0))

        surf = pygame.surface.Surface((self.width, self.height), SRCALPHA, 32)
        surf.set_alpha(128)  # surface的透明度
        U2net_result=None
        
        # Main loop
        with open(self.txtfile_path, 'a') as file:
            while True:
                for event in pygame.event.get():
                    # print(event)
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.center==[0, 0]:
                            self.center = pygame.mouse.get_pos()
                            pygame.draw.circle(self.screen, (250, 20, 15), self.center, 4)

                        elif not self.clicked and self.center!=(0,0):
                            start_pos = pygame.mouse.get_pos()

                            # self.PosiationList.append(start_pos)
                            # print(self.PosiationList)
                            self.clicked = True
                            self.drawing = True
                        else:
                            end_pos = pygame.mouse.get_pos()
                            self.PosiationX = end_pos[0]
                            self.PosiationY = end_pos[1]
                            self.radius = int(math.hypot(end_pos[0] - self.center[0], end_pos[1] - self.center[1]))
                            # self.clicked = False
                            self.drawing = False
                            self.Gray_mask=None

                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3: # (1025,3)
                        self.clicked = False
                        self.U2netON=False
                        self.center=[0, 0]
                        start_pos = (0,0)
                        surf.fill((0, 0, 0, 0))
                        self.R1=0
                        self.R2=0
                        self.R3=0
                        self.screen.blit(self.background_image, (0, 0))

                        # Image_detected=self.DetectU2net(cv2.imread("D:\\git\Deamnet\\sennsa\\20201116_151014png.png"))
                        # self.background_image = pygame.image.load(Image_detected)
                        # self.background_image = pygame.transform.scale(Image_detected, (self.width, self.height))
                        # print(event.type,event.button)
                    #pygame.event.event_name()get the string name from an event id
                    elif event.type == 770:
                        try :
                            if event.key == pygame.K_d:
                                print("pass key",pygame.KEYDOWN) 
                                self.clicked = False
                                Image_detected,self.Gray_mask=self.DetectU2net(self.IMAGE)
                                self.background_image = pygame.transform.scale(Image_detected, (self.width, self.height))
                                # print(event.type,event.button)   
                            else:
                                print("pass key",event.key )
                        except AttributeError:
                            pass               
                        
                # 2-3   測っています
                        
                if self.clicked :

                    if self.drawing == True:
                        self.screen.blit(self.background_image, (0, 0))
                        pygame.draw.line(self.screen, (255, 255, 255), self.center, start_pos, 2)

                        angle_start = math.atan2(start_pos[1] - self.center[1], start_pos[0] - self.center[0])
                        angle_end = math.atan2(pygame.mouse.get_pos()[1] - self.center[1], pygame.mouse.get_pos()[0] - self.center[0])

                        #print("math^degress",math.degrees(angle_start), math.degrees(angle_end))

                        angle_start_degrees = (math.degrees(angle_start)+ 360) % 360
                        angle_end_degrees = (math.degrees(angle_end)+ 360) % 360

                        if angle_end_degrees<angle_start_degrees:
                            angle_end_degrees=abs(360-angle_start_degrees+angle_end_degrees)
                            angle_start_degrees=0
                        else:
                            angle_end_degrees=angle_end_degrees-angle_start_degrees
                            angle_start_degrees=0

                        # 任意2个角度angle_start_degrees与angle_end_degrees，范围：[0, 360]，求其夹角的弧度的绝对值。
                        angleSE = abs(angle_start_degrees - angle_end_degrees)

                        # if angle_start_degrees-angle_end_degrees > 180:
                        #     angleSE = angle_start_degrees-angle_end_degrees
                        # else:
                        #     angleSE = 360-(angle_start_degrees-angle_end_degrees)


                        label = self.myfont.render(f"({angle_start_degrees:.2f},{angle_end_degrees:.2f})", False, (255, 255, 255))
                        label2 = self.myfont.render(f"radius = [{angleSE:.2f}]", False, (255, 255, 255))
                        self.screen.blit(label, (10, 10))
                        self.screen.blit(label2, (10, 40))

                        # pygame.draw.arc(self.screen, self.green, (self.center[0] - self.radius, self.center[1] - self.radius, self.radius * 2, self.radius * 2), -angle_end, -angle_start, 100)
                        pygame.draw.arc(surf, (0, 205, 155), (self.center[0] - self.radius, self.center[1] - self.radius, self.radius * 2, self.radius * 2), -angle_end, -angle_start, 100)
                        self.screen.blit(surf,(0,0))
                        pygame.draw.line(self.screen, (255, 255, 255), self.center, pygame.mouse.get_pos(), 2)

                    # 範囲を決まりました
                    else:  
                        # 
                        if self.tmp != (self.PosiationX, self.PosiationY):

                            # 第一次完成扇形,u2net未启动
                            if self.U2netON==False:
                                # 获得结果和灰度图
                                # print(image_path)
                                # print(len(cv2.imread(image_path,cv2.IMREAD_COLOR)))
                                _,self.Gray_mask=self.DetectU2net(self.IMAGE.copy())

                                #Gray_mask convert to pygame image
                                if self.IMAGE.shape[:2] != self.Gray_mask.shape:
                                    raise ValueError("Image and mask should have the same dimensions.")

                                rgb_new = np.zeros(self.IMAGE.shape,np.uint8)
                                rgb_new_b = self.IMAGE[:,:,2]  + 0.25*self.Gray_mask
                                rgb_new_b = rgb_new_b.astype(np.uint8)
                                rgb_new[:,:,0] = self.IMAGE[:,:,0]
                                rgb_new[:,:,1] = self.IMAGE[:,:,1]
                                rgb_new[:,:,2] = rgb_new_b

                                # 将图像数据转换为 Pygame 支持的格式
                                image_surface = pygame.image.frombuffer(rgb_new.tobytes(), (self.width, self.height), 'RGB')
                                self.background_image = pygame.transform.scale(image_surface, (self.width, self.height))

                                lineDirection = LineDirection(Image=self.Gray_mask,
                                                              Center=self.center,
                                                              Start_Pos=start_pos,
                                                              End_Pos=(self.PosiationX,
                                                                       self.PosiationY))
                                self.R1,self.R2,self.R3,Dpoint_1,Dpoint_2,Dpoint_3=lineDirection()
                                print("oLine",self.R1,self.R2,self.R3)
                                # ImageT=cv2.imread("D:\\git\Deamnet\\sennsa\\20201116_151014png.png")
                                # Image_detected=ThreadClass.ThreadWithReturnValue(target=self.DetectU2net,args=ImageT)
                                # Image_detected.start()
                                # # U2net_result = getattr(Image_detected, 'result', None)
                                # U2net_result = Image_detected.join()
                                # print(len(U2net_result))
                                # self.background_image = pygame.transform.scale(U2net_result, (self.width, self.height))
                                self.U2netON=True
                                TMPtime=datetime.now()
                                average_angle=self.average_angle(self.R1,self.R2,self.R3)
                                
                                DP=average_angle+180
                                
                                # write to file
                                try:
                                    file.write(f"{TMPtime} | AVE:[{average_angle:.2f}] | D_1:[{self.R1:.2f}] D_2:[{self.R2:.2f}] D_3:[{self.R3:.2f}] | Center:[{self.center}] Start:[{start_pos}] End:[{self.PosiationX},{self.PosiationY}]\n")
                                except Exception as e:
                                    print("can't write to file ,",e)
                            else:
                                # 释放U2net_result
                                U2net_result=None
                                pass

                            # screen update
                            self.screen.blit(self.background_image, (0, 0))
                            self.tmp = (self.PosiationX, self.PosiationY)
                            surf.fill((0, 0, 0, 0))

                            pygame.draw.line(self.screen, (255, 255, 255), self.center, start_pos, 2)
                            angle_start = math.atan2(start_pos[1] - self.center[1], start_pos[0] - self.center[0])
                            angle_end = math.atan2(self.PosiationY - self.center[1], self.PosiationX - self.center[0])

                            angle_start_degrees = (math.degrees(angle_start)+ 360) % 360
                            angle_end_degrees = (math.degrees(angle_end)+ 360) % 360

                            if angle_end_degrees<angle_start_degrees:
                                angle_end_degrees=abs(360-angle_start_degrees+angle_end_degrees)
                                angle_start_degrees=0
                            else:
                                angle_end_degrees=angle_end_degrees-angle_start_degrees
                                angle_start_degrees=0

                            # 任意2个角度angle_start_degrees与angle_end_degrees，范围：[0, 360]，求其夹角的弧度的绝对值。
                            angleSE = abs(angle_start_degrees - angle_end_degrees)

                            print("start_pos: ", start_pos, "end_pos:", end_pos, "angleSE: ", angleSE)


                            label = self.myfont.render(f"({angle_start_degrees:.2f},{angle_end_degrees:.2f})", False, (255, 255, 255))
                            label2 = self.myfont.render(f"radius = [{angleSE:.2f}]", False, (255, 255, 255))
                            label3 = self.myfont.render(f"D_1:[{self.R1:.2f}] D_2:[{self.R2:.2f}] D_3:[{self.R3:.2f}]", False, (255, 255, 255))
                            self.screen.blit(label, (10, 10))
                            self.screen.blit(label2, (10, 40))
                            self.screen.blit(label3, (10, 70))

                            pygame.draw.arc(surf, (0, 205, 155), (self.center[0] - self.radius, self.center[1] - self.radius, self.radius * 2, self.radius * 2), -angle_end, -angle_start, 100)
                            self.screen.blit(surf,(0,0))
                            # pygame.draw.arc(self.screen, self.green, (self.center[0] - self.radius, self.center[1] - self.radius, self.radius * 2, self.radius * 2), -angle_end, -angle_start, 100)
                            pygame.draw.line(self.screen, (255, 255, 255), self.center, (self.PosiationX,self.PosiationY), 2)


                            pygame.draw.line(self.screen, (255, 0, 255), self.center, Dpoint_1[1], 1) #ff00ff
                            pygame.draw.line(self.screen, (0, 255, 0), self.center, Dpoint_2[1], 1)
                            pygame.draw.line(self.screen, (255, 255, 0), self.center, Dpoint_3[1], 1) #ffff00

                            pygame.display.update()
                        else:
                            pass

                pygame.display.flip()

    def DetectU2net(self, image):
        Img,Gray_mask=self.Tester.detectU2net(image)
        
        #cv2.imwrite(r'D:\git\Deamnet\test_data\index_20240124\o0115mask.png', Gray_mask) 

        return pygame.image.frombuffer(Img.tobytes(), (self.width, self.height),"RGB"),Gray_mask
    
    # not used
    def get_angle(Px, Py, Cx, Cy):
        '''
        Calculate the angle of the line with respect to the 12 o'clock position
        '''
        dx = Px - Cx
        dy = Py - Cy 
        angle = (np.arctan2(dy, dx) * 180.0 / np.pi) % 360  # Convert radian to degree and ensure positive angle
        # 90-angle
        angle_from_12 = (angle+90) % 360  # Adjust angle to the 12 o'clock reference
        print("angle_from_12:",angle_from_12)
        return angle_from_12
    
    # not used
    def ArrowDirection(self, image):
        """
        Find the angle of the line with respect to the 12 o'clock position
        """
        # Get the image center
        image_center = self.center

         # Find the largest contour and its convex hull and approximate polygon
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

        # Find the farthest point from the center on the convex hull
        farthest_dist = 0
        farthest_point = None
        for point in approx[:, 0, :]:
            dist = np.linalg.norm(point - image_center)
            if dist > farthest_dist:
                farthest_dist = dist
                farthest_point = tuple(point)

        # Draw the line from the image center to the farthest point
        cv2.line(image_with_hull, image_center, farthest_point, (255, 0, 0), 2)

        # Calculate the angle of the line with respect to the 12 o'clock position
        angle_from_12 = self.get_angle(farthest_point[0], farthest_point[1], image_center[0], image_center[1])

        cv2.imwrite(r'D:\git\Deamnet\index_2023_ver2_result0115\o02161a.png', cv2.cvtColor(image_with_hull, cv2.COLOR_BGR2RGB)) 

    # take 3 angle of average
    def average_angle(self, angle_1, angle_2, angle_3):
        numbers=[angle_1, angle_2, angle_3]
        pairs = [(a, b, abs(a - b)) for a in numbers for b in numbers if a != b]
        closest_pair = min(pairs, key=lambda x: x[2])

        return (closest_pair[0] + closest_pair[1]) / 2

# -------button---------
# 将二进制图像转换为灰度图
# gray_point_mask = (point_mask * 255).astype(np.uint8)

# # 保存灰度图到根目录
    
# cv2.imwrite("gray_point_mask.png", gray_point_mask) 
#-----------------------

if __name__ == '__main__':
    tester = Tester()
    image_path="/var/pythonlibtmp/U2netIndex04/data/20240123_133412.jpeg"
    # image_path="C:\\Users\\seiko\\Downloads\\gicg\\image20240122\\image\\0x00000000_img_20240120073226688.jpg"
    # image_path="C:\\Users\\seiko\\Downloads\\gicg\\image20240122_平原_0x00\\image\\0x00000000_20240121100252623_img.jpg"
    indexveriewer = IndexVeiwer(tester,image_path)
    indexveriewer(image_path=image_path)


