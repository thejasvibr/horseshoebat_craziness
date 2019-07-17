# -*- coding: utf-8 -*-
""" Graphical user interface to run the timestamp checking and 
Created on Wed Jul 17 17:22:44 2019

@author: tbeleyur
"""

import cv2 
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

video_path = 'video/DVRecorder_03_20190704_16.49.45-16.56.42[R][@da37][0].avi'

plt.figure()
a0 = plt.subplot(111)




class InteractivePlot():
    def __init__(self):
        self.index = 0
    
    def load_video(self):
        self.video = cv2.VideoCapture(self.video_path)
        self.plot_image()

    def plot_image(self):
        self.video.set(1,self.index)
        success, frame = self.video.read()
        a0.imshow(frame)
        plt.xticks([])
        plt.yticks([])
        a0.set_title('Frame number: '+str(self.index))
    
    def move_forward(self,event):
        self.index +=1
        self.plot_image()

    def move_backward(self,event):
        self.index -= 1 
        self.plot_image()
    
    def move_to(self, box_input):
        self.index = int(box_input)
        self.plot_image()
    
    
ip = InteractivePlot()
ip.video_path = video_path
ip.load_video()


axnext = plt.axes([0.81,0.05,0.1,0.075])
axprev = plt.axes([0.65,0.05,0.1,0.075])
framenum = plt.axes([0.45,0.05,0.1,0.075])
move_to = TextBox(framenum, 'Move To:')
move_to.on_submit(ip.move_to)
next_button = Button(axnext,'NEXT FRAME')
prev_button = Button(axprev,'PREV FRAME')
next_button.on_clicked(ip.move_forward)
prev_button.on_clicked(ip.move_backward)


