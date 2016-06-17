
# coding: utf-8

# # CapStone Project
# 
# #### This project will try to create a convolution Neural network, train it with Street view house numbers and calculate the accuracy of predicting of house numbers 
# 
# Data set 
# http://ufldl.stanford.edu/housenumbers/
# 
# ##### For visualization of the house numbers images you can have a look at the site provided above

# In[27]:

import tensorflow as tf
import scipy.io as scp
import numpy as np
import random


# In[28]:

# Load the data 
testData = scp.loadmat('../../data/svhn/test_32x32.mat')
trainData = scp.loadmat('../../data/svhn/train_32x32.mat')


# In[29]:

testDataX = testData['X']
testDataY = testData['y']

trainDataX = trainData['X']
trainDataY = trainData['y']


# In[30]:

print type(trainDataX)
print type(trainDataY)

print 'train Data image shape : ', trainDataX.shape
print 'train data output shape : ', trainDataY.shape
print 'test data image shape : ', testDataX.shape
print 'test data output shape : ', testDataY.shape


# In[31]:

# try tansposing the array
def transposeArray(data):
    xtrain = []
    trainLen = data.shape[3]
    for x in xrange(trainLen):
        xtrain.append(data[:,:,:,x])
    
    xtrain = np.asarray(xtrain)
    return xtrain


# In[32]:

trainDataX = transposeArray(trainDataX)
testDataX = transposeArray(testDataX)


print 'New train data image shape : ', trainDataX.shape


# In[33]:

def OnehotEndoding(Y):
    Ytr=[]
    for el in Y:
        temp=np.zeros(10)
        if el==10:
            temp[0]=1
        elif el==1:
            temp[1]=1
        elif el==2:
            temp[2]=1
        elif el==3:
            temp[3]=1
        elif el==4:
            temp[4]=1
        elif temp[5]==1:
            temp[5]=1
        elif temp[6]==1:
            temp[6]=1
        elif temp[7]==1:
            temp[7]=1
        elif temp[8]==1:
            temp[8]=1
        elif temp[9]==1:
            temp[9]=1
        Ytr.append(temp)
        
    return np.asarray(Ytr)


# In[34]:

# convert y to one hot encoding
trainDataY = OnehotEndoding(trainDataY)
testDataY = OnehotEndoding(testDataY)
print 'train data output shape : ', trainDataY.shape
print 'test data output shape : ', testDataY.shape


# In[35]:

#Neural network parameters
height = 32
width = 32
channel = 3
tags = 10
patch = 5
depth = 16
num_hidden = 64
dropout = 0.75 # Dropout, probability to keep units

learning_rate = 0.00001


# In[36]:

tf_X = tf.placeholder("float", shape=[None, height, width, channel])
tf_Y = tf.placeholder("float", shape=[None, tags])

convW1 = tf.Variable(tf.random_normal([patch, patch, channel, depth], stddev=0.1))
bias1 = tf.Variable(tf.random_normal([depth], stddev=0.1))

convW2 = tf.Variable(tf.random_normal([patch, patch, depth, depth], stddev=0.1))
bias2 = tf.Variable(tf.random_normal([depth], stddev=0.1))

w3 = tf.Variable(tf.random_normal([height // 4 * width // 4 * depth, num_hidden], stddev=0.1))
bias3 = tf.Variable(tf.random_normal([num_hidden]))

w4 = tf.Variable(tf.random_normal([num_hidden, tags], stddev=0.1))
bias4 = tf.Variable(tf.random_normal([tags], stddev=0.1))  

keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# ### Here is the model I have tried to build
# Input Image : 32x32x3
# first Convolution Hidden layer : 5x5x3x16
# Padding : Same, Stride : [1,2,2,1]
# Output of first Convolution Hidden layer : 16x16x16
# 
# Second Convolution Hidden layer : 5x5x16x16
# Padding : Same, Stride : [1,2,2,1]
# Output of Second Convolution Hidden layer : 8x8x16
# 
# third Hidden Layer fully connected : 8x8x16
# Output of third Hidden layer : 64
# 
# Fourth Hidden Layer : 64 x 10

# In[37]:

#model

def model(X):
    
    #first layer : Convolution
    conv = tf.nn.conv2d(X, convW1, [1,1,1,1], padding='SAME')
    hidden1 = tf.nn.relu(conv + bias1)
    
    #second layer : pooling
    hidden2 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    #third layer : convolution
    conv2 = tf.nn.conv2d(hidden2, convW2, [1,1,1,1], padding='SAME')
    hidden3 = tf.nn.relu(conv2 + bias2)
    
    #fourth layer : pooling
    hidden4 = tf.nn.max_pool(hidden3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    #reshape it to a single Dimensional
    shape = hidden4.get_shape()
    
    #5th layer : fully connected
    newInput = tf.reshape(hidden4, [-1, shape[1].value * shape[2].value * shape[3].value])
    hidden5 = tf.nn.relu(tf.matmul(newInput, w3) + bias3)
    
    dp5 = tf.nn.dropout(hidden5, keep_prob)
    
    return tf.matmul(dp5, w4) + bias4


# In[38]:

pred = model(tf_X)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, tf_Y))

# Optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(tf_Y,1)), "float"))


# In[ ]:

def Accuracy(X, Y, message, sess):    
    print message, sess.run(accuracy, feed_dict= {tf_X: X, tf_Y: Y, keep_prob:1.0})


# In[ ]:

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    epoch = 10000
    batch_size = 500
    print('Initialized')
    
    p = np.random.permutation(range(len(trainDataX)))
    trX, trY = trainDataX[p], trainDataY[p]
    
    for step in range(epoch):
        
        batch = np.random.choice(len(trainDataX) - 1, batch_size)
        inX, outY = trainDataX[batch], trainDataY[batch]
        sess.run(optimizer, feed_dict= {tf_X: inX, tf_Y: outY, keep_prob:0.75})
        
        if step % 500 == 0:
            print 'cost at each step :', step, 'is :', sess.run(loss, feed_dict={tf_X: inX, tf_Y: outY, keep_prob:1.0})
    
    #Accuracy(trX, trY, 'accuracy of training data : ', sess)
    Accuracy(testDataX, testDataY, 'accuracy of test data : ', sess)


# In[ ]:



