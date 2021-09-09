#!/usr/bin/env python3
"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import rospy
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

from sensor_msgs.msg import Image as SensorMsgImage
from minau.msg import SonarTargetList, SonarTarget
from math import pi

def grad_to_rad(grads):
    return 2 * pi * grads / 400


class Detector:
    def __init__(self,
        weights='best.pt',  # model.pt path(s)
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        num_grads = 40,
        image_path = "",
        sonar_topic = "",
        sonar_range = 10.0
    ):
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.classes = classes
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.augment = augment
        self.visualize = visualize


        # Initialize
        set_logging()
        self.device = select_device(device)

        # Load model
        w = weights[0] if isinstance(weights, list) else weights
        stride, self.names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
          

        imgsz = check_img_size(imgsz, s=stride)  # check image size
        self.num_grads = num_grads

        self.range = 10.0
        self.detection_pub = rospy.Publisher("sonar_processing/target_list",SonarTargetList,queue_size=10)
        rospy.Subscriber(sonar_topic, SensorMsgImage, self.image_callback)

    def image_callback(self, msg):
        print("Recieved Image")
        np_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        im = Image.fromarray(np_img)
        im.save("temp.png")
        stride = int(self.model.stride.max())
        dataset = LoadImages("temp.png", img_size=self.imgsz, stride=stride)
        for path, img, im0s, vid_cap in dataset:
            im0 = im0s.copy()

        # cv2.imshow("image", np_img)
        # cv2.waitKey()
        
            img = torch.from_numpy(np.array(img)).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            # print(img)

            # Inference
            t1 = time_sync()
            
            pred = self.model(img, augment=self.augment, visualize=self.visualize)[0]

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            t2 = time_sync()
            # return
            # Process predictions
            for i, det in enumerate(pred):  # detections per image
                s = ''
                targets = []
                for j in range(len(det)):
                # if len(det):
                    # Rescale boxes from img_size to im0 size
                    print("detection: ", end="")
                    print(det)
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    print("detection: ", end="")
                    print(det[0])
                    frame_grad_angle = float(msg.header.frame_id)

                    detection_x = ((det[j][0] + det[j][2])/2.0)
                    detection_y = ((det[j][1] + det[j][3])/2.0) - 10

                    detection_range = detection_x * self.range / msg.width

                    image_angle = detection_y * self.num_grads / msg.height
                    total_grad_angle = frame_grad_angle + 20 - image_angle
                    rad_angle = grad_to_rad(total_grad_angle)

                    while rad_angle > np.pi:
                        rad_angle -= 2*np.pi

                    while rad_angle < -np.pi:
                        rad_angle += 2*np.pi

                    targets.append(SonarTarget("Detection", rad_angle, 0.1, 0, 0.1, detection_range, 0.1, False, 
                        SonarTarget().TARGET_TYPE_UNKNOWN, SonarTarget().UUV_CLASS_UNKNOWN))

                if len(targets):
                    header = msg.header
                    header.frame_id = "base_link"
                    stl = SonarTargetList(header, targets)
                    self.detection_pub.publish(stl)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                


if __name__ == "__main__":
    rospy.init_node("Yolo_Detector")

    # Load in rosparams
    weights = rospy.get_param("~weights")
    image_path = rospy.get_param("~image_path") 
    sonar_topic = rospy.get_param("~sonar_topic")
    num_grads = rospy.get_param("~num_grads")
    iou_thres = rospy.get_param("~iou_thres")
    conf_thres = rospy.get_param("~conf_thres")
    sonar_range = rospy.get_param("~sonar_range")
    
    
    # check_requirements(exclude=('tensorboard', 'thop'))
    Detector(weights=weights, conf_thres=conf_thres, iou_thres=iou_thres, num_grads=num_grads, 
        image_path=image_path, sonar_topic=sonar_topic, sonar_range=sonar_range)

    while not rospy.is_shutdown():
        rospy.spin()
