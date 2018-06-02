#####
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

#import caffe
caffe_root = './caffe/' 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

mean_file ='./mean.binaryproto'
proto_data = open(mean_file, "rb").read()
#print(proto_data)
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]

#Load gender pretrained model
gender_pretrained_model ='./gender_net.caffemodel'
gender_model_file='./gender_model/deploy.prototxt'
gender = caffe.Classifier(gender_model_file, gender_pretrained_model,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

#labels for gender prediction
gender_label=['Male','Female']

#Testing (input from user)
test_image = './Example/12.jpg'

#Load image to caffe model
input_image = caffe.io.load_image(test_image)
plt.imshow(input_image)

#make_prediction
prediction = gender.predict([input_image]) 
print ('predicted gender:', gender_label[prediction[0].argmax()])
