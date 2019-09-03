
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import scipy as sc
import h5py
import time
import matplotlib.pyplot as plt
import math
import seaborn as sns


# In[2]:


MessageWordsArray = np.array([
	[ 0, 0, 0, 0],
	[ 0, 0, 0, 1],
	[ 0, 0, 1, 0],
	[ 0, 0, 1, 1],
	[ 0, 1, 0, 0],
	[ 0, 1, 0, 1],
	[ 0, 1, 1, 0],
	[ 0, 1, 1, 1],
	[ 1, 0, 0, 0],
	[ 1, 0, 0, 1],
	[ 1, 0, 1, 0],
	[ 1, 0, 1, 1],
	[ 1, 1, 0, 0],
	[ 1, 1, 0, 1],
	[ 1, 1, 1, 0],
	[ 1, 1, 1, 1]])

MessageWordsArray_test = np.array([
	[ 0, 0, 0, 0],
	[ 0, 0, 0, 1],
	[ 0, 0, 1, 0],
	[ 0, 0, 1, 1],
	[ 0, 1, 0, 0],
	[ 0, 1, 0, 1],
	[ 0, 1, 1, 0],
	[ 0, 1, 1, 1],
	[ 1, 0, 0, 0],
	[ 1, 0, 0, 1],
	[ 1, 0, 1, 0],
	[ 1, 0, 1, 1],
	[ 1, 1, 0, 0],
	[ 1, 1, 0, 1],
	[ 1, 1, 1, 0],
	[ 1, 1, 1, 1]])


# In[3]:


def AddNoise(std_dev,l): #This function adds soft noise based on a value on rand values, up measures the upper limit of noise, l measures length of array that takes noise
    return(np.random.normal(0,std_dev,l))


# In[4]:


def RepetitionCodes(x,n): # n is the number of times you want to repeat the code and i is the number of times you do
    temp = x
    for i in range(n-1):    
        x = np.append(x,temp)
    return(x)


# In[5]:


def ber(y_true, y_pred):
    return(tf.reduce_mean(tf.cast(tf.not_equal(tf.round(y_true), tf.round(y_pred)),dtype='float64')))


# In[6]:


print(ber([0,0,0],[1,0,0]))


# In[7]:


repitions = 8
input_vector_length = 4*repitions
hidden_layer_nodes = 4
output_vector_length = 4
learning_rate = 1e-3
training_points = input_vector_length


# In[8]:


#This is to define the input and desired outputs
Input_codes = tf.placeholder(tf.float64,[None, input_vector_length],name = 'input') #This is the vector that is sent into the NN
Output_desired_codes = tf.placeholder(tf.float64,[None, output_vector_length], name = 'output_desired') #This is the desired output on the codes


# In[9]:


#This is to define the transition matrices(with biases) and activations
#W1 = tf.Variable(tf.random_normal([input_vector_length,output_vector_length],stddev=0.3,dtype ="float64"),name = "Input_Layer", dtype ="float64")
B = tf.Variable(tf.random_normal([1,output_vector_length],stddev=0.3,dtype = "float64"),name ="Input_Layer_biases", dtype = "float64")
W1 = tf.Variable(tf.random_normal([input_vector_length,output_vector_length],stddev=0.3,dtype = "float64"),name = "Input_Layer", dtype ="float64")


# In[10]:


#Output layer is defined here
Output_layer = tf.sigmoid(tf.add(tf.matmul(Input_codes,W1),B))
BER = ber(Output_layer,Output_desired_codes)


# In[11]:


#Cost function and optimizer
mse = tf.reduce_mean(tf.square(Output_layer - Output_desired_codes))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse)


# In[12]:


#Setup initialization
init_op = tf.global_variables_initializer()


# In[13]:


#Adding Noise to the input vectors here
coded_train = np.zeros([len(MessageWordsArray),input_vector_length])
for i in range(0,len(MessageWordsArray)):
    coded_train[i,:] = RepetitionCodes(MessageWordsArray[i,:],repitions) + AddNoise(0.3,input_vector_length)
coded_validation = np.zeros([len(MessageWordsArray),input_vector_length])
for i in range(0,len(MessageWordsArray)):
    coded_validation[i,:] = RepetitionCodes(MessageWordsArray[i,:],repitions) + AddNoise(0.5,input_vector_length)
coded_test = np.zeros([len(MessageWordsArray_test),input_vector_length])
#for i in range(0,len(MessageWordsArray_test)):
 #   coded_test[i,:] = RepetitionCodes(MessageWordsArray_test[i,:]) + AddNoise(0.6,input_vector_length)


# In[19]:


epochs = 10000
batch_size = 32
with tf.Session() as sess: #Start the session
    #Intialise the variables
	writer = tf.summary.FileWriter("./logs/log_repition")
	writer.add_graph(sess.graph)
	sess.run(tf.global_variables_initializer())
	total_batch = int(training_points/batch_size) # This is used to calculate the average loss in each iteration
	x , y = [],[]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	fig.show()
    #print(total_batch)
	for j in range(100):
		sess.run(tf.global_variables_initializer())
		for epoch in range(epochs):
			average_loss = 0 #initialize average_loss as zero
			for i in range(total_batch):
	            #x_batch, y_batch = y_train, x_train
				_,c,w1,b,ber_final,output_layer = sess.run([optimizer,mse,W1,B,BER,Output_layer],feed_dict = {Input_codes: coded_train, Output_desired_codes: MessageWordsArray})
				average_loss = c/total_batch
	            #print(w1,w2,c,_)
	#             for g, v in gradients_and_vars:
	#                 if g is not None:
	#                     print "****************this is variable*************"
	#                     print "variable's shape:", v.shape
	#                     print v
	#                     print "****************this is gradient*************"
	#                     print "gradient's shape:", g.shape
	#                     print g
	            #print(w1,w2)
	            #sess.run(assign_op)
	            #sess.run(assign_op_w1)
	            #print(iteration.value())
	            #W1 = tf.add(W1,tf.multiply(tf.random_normal([input_vector_length,hidden_layer_nodes],stddev=0.1),tf.math.pow(alpha,tf.constant(i,'float32'))))
				#print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(average_loss),"Bit Error rate = ",ber_final)
			mse_test = sess.run(mse, feed_dict={Input_codes: coded_validation, Output_desired_codes:MessageWordsArray})
		print(j)
		x.append(j)
		y.append(mse_test)
	ax.plot(x,y,color = 'b')
	fig.canvas.draw
	plt.savefig('NotTooMuchNoiseTrainErrorData_03')


# In[22]:


print(w1,b)

