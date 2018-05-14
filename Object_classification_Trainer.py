
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
import tflearn
import tensorflow as tf
import pickle


# In[2]:


data = tflearn.datasets.cifar10.load_data(one_hot=True)


# In[3]:


tflearn.init_graph(num_cores=4,gpu_memory_fraction=0.5)


# In[4]:


net = tflearn.input_data(shape=[None,32,32,3])
net = tflearn.layers.conv_2d(net,32,3,activation='relu')
net = tflearn.layers.max_pool_2d(net,2)
net = tflearn.layers.conv_2d(net,64,3,activation='relu')
net = tflearn.layers.conv_2d(net,64,3,activation='relu')
net = tflearn.layers.max_pool_2d(net,2)
net = tflearn.layers.fully_connected(net,512,activation='relu')
net = tflearn.layers.dropout(net,0.5)
net = tflearn.layers.fully_connected(net,10,activation='softmax')
net = tflearn.layers.estimator.regression(net,optimizer='adam',loss='categorical_crossentropy',
                     learning_rate=0.001)


# In[5]:


(x,y),(x_test,y_test) = data


# In[6]:


with tf.device("/gpu:1"):
    model = tflearn.DNN(net,checkpoint_path='model/object_classifier.ckpt')
    model.fit(x,y,n_epoch=50,shuffle=True,validation_set=(x_test,y_test),batch_size=96)
    model.save('model/model.tfl')


# In[9]:




