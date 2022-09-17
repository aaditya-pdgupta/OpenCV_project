# Introduction

Computer vision is a field of artificial intelligence (AI) that enables computers and systems to derive meaningful information from digital images, videos, and other visual inputs — and take actions or make recommendations based on that information. It plays a significant role in self-driving cars, robotics, and photo correction apps. OpenCV (Open Source Computer Vision Library) is an open-source library mainly aimed at real-time computer vision and image processing. Originally developed by Intel, it was later supported by Willow Garage then Itseez (which was later acquired by Intel).

Nowadays, the use of AI has a broad area which is facialiated by the openCV. The openCV can be used for the object detection and face recognition in real time based.  Besides face recognition and object recognition, there are lots of applications that are solved using OpenCV. Some of them are automated inspection and surveillance such as vehicle counting on highways along with their speeds,  anomaly (defect) detection in the manufacturing process (the odd defective products), interactive art installations, video/image search and retrieval, robot and driver-less car navigation and control, medical image analysis, Movies – 3D structure from motion, TV Channels advertisement recognition, motion understanding, structure form motion, augmented reality, boosting , decision tree learning, Artificial Neural Network (ANN), etc. Here, we will discuss about face recognition and object detection below:

# Object Detection

Object Detection is a computer technology related to computer vision, image processing, and deep learning that deals with detecting instances of objects in images and videos. Object vision is also known as an important computer vision task used to detect instances of visual objects of certain classes (for example humans, animals, cars or buildings) in digital images such as photos or video frames. The goal of object detection is to develop computional models that provide the most fundamental information neded by the computer vision applications. 

This project detects objects using the mobileNet SSD method. Therefore before proceeding, three files are a pre-requisite_'coco.names', 'ssd_mobolenet_v3_large_coco_2020_01_14.' and 'frozen_inference_graph.pb'. The code is saved in Object_detection for the program. This code when run, will detect objects provided the 3 files mentioned before are stored in the same folder as the python file.  For a deep learning or machine learning model we need good orecision. Here coco.names comes into picture. COCO stands for Common Objects in Context.  This dataset contains objects from an everyday context. COCO dataset provides the labeling and segmentation of the objects in the images. There are 80 object categories of labeled and segmented images in the file. Thus our model can detect and identify 80 types of objects in this code.

Here, we have used className to retrieve the 80 object categories 0f coco.names so that we can use them. This is where we can use file handling and string methods. We read our file using f.read(), remove or strip the spaces on the right of words using .rstrip(\n) and separated words by breaking lines where there is space using split(\n). Refer to string methods to get a proper grasp on these concepts. Here, file 'sd_mobilenet_v3_large_coco_2020_01_14.pbtxt' is a Single-Shot multibox Detection (SSD) network , intended to perform object detection. In simple words this file is a pre-trained Tensorflow model and has already been trained on the COCO dataset. 

The file 'frozen_inference_graph.pb' is a about frozen graph. Freezing is the process to identify and save all of required things (grapg, weights etc) in a single file that you can easily use. Frozen graphs are commonly used for inference in Tensorflow and are stepping stones for inference for other frameworks. The b in bbox is for bounding in the program. Bounding boxes can be useful as as standalone shapes, but they are primarily used for approximating more complex shapes to speed operations. Zip()  and .flatten() in the program makes a set/list from the values taken by its arguments. The purpose of Python zip() method is to map the similar index of multiple containers so that they can be used just using as single entity. Flattening lists in python means converting multidimensional lists into one-dimensional lists. It is basically a method of merging all the sublists of a nested list into one unified list. In openCV rectangle() function is used to draw  a rectangle around the dedicated box and putText function put the object name on top of the rectangle using the COCO names. The imread() function read the image where as imshow show the image in the output.  In program threshold is used to create binary images ans plt.savefig save the output.

## Input figures

<p align="center">
  <img src="../main/Object_detection/street.jpg"  width="48%" />
  <img src="../main/Object_detection/cycle_detection.png"  width="48%" /> 
</p>

## Output figures

<p align="center">
  <img src="../main/Object_detection/street_detection_2.png"  width="48%" />
  <img src="../main/Object_detection/cycle_detection_2.png"  width="48%" /> 
</p>

# Face recognition

Face detection is a computer vision technology that helps to locate/visualize human faces in digital images. This technique is a specific use case of object detectiom technology that deals with detecting instances of semantic objects of certain class (such as jumans, buildings or cars) in digital images and videos. With the advent of technology, face detection has gained a lot of importance especially in fields such as photography, security,marketing, etc. 

In order to work, face detection applications use machine learning algorithms to detect the human faces within images of any size. The larger images might contain numerous objects that are not facing such as landscapes, objects, animals, buildings and other parts of humans (eg. legs, shoulder and arms). Facial detection technology was previously associated with only security sector but today there is active expansion into other industries including retail, healthcare etc.

In the program we have used a CascadeClassifier which is a classifier used to detect the object for which it has been trained for. The Haar cascade is trained by superimposing the positive image over a set of images. This type of training is generally done a server and on various stages. Better results are obtained by using high quality images and increasing the amount of stages for which the classifier is trained for. OpenCV comes with lots of pre-trained classifiers. Those XML files can be loaded by cascadeClassifier method of the cv2 module. Here we have used haarcascade_frontalface_default.xml for detecting faces. Initially, the image is a three-layer image (i.e., RGB), So It is converted to a one-layer image (i.e., grayscale). The face detection method on the grayscale image is applied in the program. 

This is done using the cv2::CascadeClassifier::detectMultiScale method, which returns boundary rectangles for the detected faces (i.e., x, y, w, h). It takes two parameters namely, scaleFactor and minNeighbors. ScaleFactor determines the factor of increase in window size which initially starts at size “minSize”, and after testing all windows of that size, the window is scaled up by the “scaleFactor”, and the window size goes up to “maxSize”. If the “scaleFactor” is large, (e.g., 2.0), there will be fewer steps, so detection will be faster, but we may miss objects whose size is between two tested scales. (default scale factor is 1.3). Higher the values of the “minNeighbors”, less will be the number of false positives, and less error will be in terms of false detection of faces. However, there is a chance of missing some unclear face traces as well. Rectangles are drawn around the detected faces by the rectangle method of the cv2 module by iterating over all detected faces. 

## Input figure

<p align="center">
  <img src="../main/Face_recogination/face_detection.jpg" width="250" height="250"/>
</p>
