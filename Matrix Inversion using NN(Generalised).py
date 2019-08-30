
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
from scipy.linalg import orth
from scipy.linalg import block_diag


Gamma = 0 #Parameter to deviate orthogonal vectors slightly
#input_vectors = 10*(np.random.rand(input_vector_length,training_points) - 0.5) #Generate required length of input vectors
#input_vectors = orth(input_vectors) + Gamma*np.random.randn(input_vector_length,training_points) #Orthogonalise them and add some random noise
input_vectors = np.array([[10,0],[10,0.05]]) #If you want to initalize your own training set
#W = np.array([[0.20774353,1.0305219],[-1.2163291,-0.1880631]])
#orthonormal_vectors = np.matmul(W,orthonormal_vectors)
#print(input_vectors)

Regularization_1 = tf.constant([10,0], dtype = "float64",shape = [2,1])
Regularization_2 = tf.constant([10,0.05], dtype ="float64", shape = [1,2])

input_vector_length = 2
hidden_layer_nodes = 2
output_vector_length = 2
learning_rate = 1e-2
training_points = input_vector_length
iteration = tf.Variable(1.1,name = 'iteration', dtype = "float64") #Used for the weight decay upgrade
#print(iteration)
#updater = tf.constant(1)
#iteration = tf.add(iteration,updater) 
assign_op = tf.assign(iteration,iteration + 1) # This is for incrementing it every time
alpha = tf.constant(0.9999,dtype = 'float64')

# In[3]:

#Initialize placeholders , which are variable shape empty objects into which any size tensor can be inputed
Input_layer = tf.placeholder(tf.float64, [None,input_vector_length],name = 'input')
#Output_layer = tf.placeholder(tf.float32, [None,output_vector_length],name = 'output')
Output_vectors = tf.placeholder(tf.float64, [None,output_vector_length],name = 'labels')


# In[4]:


#Weights for the hidden layer and biases, Biases not needed for this particular problem
# for input to hidden layer
#W1 = tf.Variable(tf.random_normal([input_vector_length,hidden_layer_nodes],stddev=0.1), name='W1')
W1 = tf.Variable([[0.5,0.4],[0.3,0.2]], name = 'W1',dtype = "float64")
# for hidden to output layer
#W2 = tf.Variable(tf.random_normal([hidden_layer_nodes,output_vector_length],stddev=0.1), name='W2')
W2 = tf.Variable([[0.3,0.4],[0.7,0.1]], name = 'W2',dtype = "float64")
#The problem here was I used MSE in a terrible way
#hidden_layer_1 = tf.contrib.layers.fully_connected(Input_layer, output_vector_length, None,biases_regularizer=)


# In[5]:


#Create the operations on the hidden layer
hidden_layer_1 = tf.matmul(Input_layer,W1)


# In[6]:


#Create operations for the output layer
Output_layer = tf.matmul(hidden_layer_1,W2)


# In[7]:
#print(np.matmul(np.array([0.5,0.4],[0.3,0.2]),input_vectors[0]))

#Create Loss function
mse_real = tf.reduce_mean(tf.square(Output_layer - Output_vectors))
mse = tf.multiply(1-(alpha**iteration),tf.reduce_mean(tf.square(Output_layer - Output_vectors))) + tf.multiply(alpha**iteration,tf.matmul(tf.linalg.matmul(Regularization_2,tf.transpose(W1)),tf.linalg.matmul(W1,Regularization_1)))

# In[8]:


#The Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse_real)
#grads_and_vars = optimizer.compute_gradients(mse)
#assign_op_w1 = tf.assign(W1,W1 + (alpha**iteration)*tf.random_normal([input_vector_length,hidden_layer_nodes],stddev=0.1))
#optimizer_real = optimizer.apply_gradients(grads_and_vars)


# In[9]:


#Setup initialization
init_op = tf.global_variables_initializer()


# In[10]:


#Code to find angle b/w two vectors
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))
def length(v):
  return math.sqrt(dotproduct(v, v))
def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


# In[11]:





# In[12]:


#Ax = y, We need to invert A => x_train is actually the output of the NN while y_train is the input
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
A = 10*np.random.randn(input_vector_length,input_vector_length) #Generate a random matrix to invert
x_train = np.matmul(np.linalg.inv(A),input_vectors) # use the generated A and input_vectors to generate x (See Ax = y)
#x_validate = np.random.randn(output_vector_length) # Generate a dingle vector for validation
x_validate = np.array([1,2])
x_train = np.transpose(x_train) #Transpose this for right multiplication
y_train = input_vectors #Set y_train(The one we'll be sending to the feed forward NN) as input_vectors that were initially generated
y_train = np.transpose(y_train) #Keep x_train and y_train consistent
y_validate = np.reshape((np.matmul(A,x_validate)),(1,output_vector_length))#Appropraite shape for the NN
x_validate = np.reshape(x_validate,(1,output_vector_length))#Appropriate shape for the NN


# In[13]:


#print(x_train)
#print(np.shape(x_train))
#print(y_train)
#print(np.shape(y_train))
#print(x_validate)
#print(np.shape(x_validate))
#print(y_validate)
#print(np.shape(y_validate))


# In[17]:


#Some parameters
epochs = 100000
batch_size = 2

# In[ ]:

#print(grads_and_vars)

with tf.Session() as sess: #Start the session
    #Intialise the variables
    writer = tf.summary.FileWriter("./logs/log_inv")
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    total_batch = int(training_points/batch_size) # This is used to calculate the average loss in each iteration
    #print(total_batch)
    for epoch in range(epochs):
        average_loss = 0 #initialize average_loss as zero
        for i in range(total_batch):
            #x_batch, y_batch = y_train, x_train
            _,c,w1,w2 = sess.run([optimizer,mse_real,W1,W2],feed_dict = {Input_layer: y_train, Output_vectors: x_train})
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
            sess.run(assign_op)
            #sess.run(assign_op_w1)
            #print(iteration.value())
            #W1 = tf.add(W1,tf.multiply(tf.random_normal([input_vector_length,hidden_layer_nodes],stddev=0.1),tf.math.pow(alpha,tf.constant(i,'float32'))))
            
            print(sess.run(mse, feed_dict={Input_layer:y_validate , Output_vectors: x_validate}))
        print("Epoch:", (epoch + 1), "cost =", average_loss)



# In[19]:


print(w1)

#tf.summary

# In[18]:


print(np.linalg.inv(A))

