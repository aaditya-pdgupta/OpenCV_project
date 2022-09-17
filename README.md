# Introduction

Computer vision is a field of artificial intelligence (AI) that enables computers and systems to derive meaningful information from digital images, videos, and other visual inputs — and take actions or make recommendations based on that information. It plays a significant role in self-driving cars, robotics, and photo correction apps. OpenCV (Open Source Computer Vision Library) is an open-source library mainly aimed at real-time computer vision and image processing. Originally developed by Intel, it was later supported by Willow Garage then Itseez (which was later acquired by Intel).

Nowadays, the use of AI has a broad area which is facialiated by the openCV. The openCV can be used for the object detection and face recognition in real time based.  Besides face recognition and object recognition, there are lots of applications that are solved using OpenCV. Some of them are automated inspection and surveillance such as vehicle counting on highways along with their speeds,  anomaly (defect) detection in the manufacturing process (the odd defective products), interactive art installations, video/image search and retrieval, robot and driver-less car navigation and control, medical image analysis, Movies – 3D structure from motion, TV Channels advertisement recognition, motion understanding, structure form motion, augmented reality, boosting , decision tree learning, Artificial Neural Network (ANN), etc. Here, we will discuss about face recognition and object detection below:

# Object Detection

Object Detection is a computer technology related to computer vision, image processing, and deep learning that deals with detecting instances of objects in images and videos. Object vision is also known as an important computer vision task used to detect instances of visual objects of certain classes (for example humans, animals, cars or buildings) in digital images such as photos or video frames. The goal of object detection is to develop computional models that provide the most fundamental information neded by the computer vision applications. 

This project detects objects using the mobileNet SSD method. Therefore before proceeding, three files are a pre-requisite_'coco.names', 'ssd_mobolenet_v3_large_coco_2020_01_14.' and 'frozen_inference_graph.pb'. The code is saved in Object_detection for the program. This code when run, will detect objects provided the 3 files mentioned before are stored in the same folder as the python file.  For a deep learning or machine learning model we need good orecision. Here coco.names comes into picture. COCO stands for Common Objects in Context.  This dataset contains objects from an everyday context. COCO dataset provides the labeling and segmentation of the objects in the images. There are 80 object categories of labeled and segmented images in the file. Thus our model can detect and identify 80 types of objects in this code.

