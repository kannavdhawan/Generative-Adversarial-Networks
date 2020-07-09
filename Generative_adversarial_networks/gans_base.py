import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf

def create_y(x):

    return 10+x*x

def sample(num,scaling):    
    """
    Creating quadratic data.
    Atrb: 
        num: Number of samples
        scaling: Norm to scale the randomly generated sample.
        
    Returns: numpy array with x and y.
    """

    data_set=[]
    x=scaling*(np.random.random_sample((num,))-0.5)

    for i in range(num):
        y_i=create_y(x[i])
        data_set.append([x[i],y_i])
    data_set=np.asarray(data_set)
    return data_set
    
data_set=sample(10000,100) 
print(data_set)
# print(data_set.shape)

data_set_df=pd.DataFrame(data_set)
# plt.plot(data_set_df[0],data_set_df[1],'o',linewidth=2, markersize=2)
sns.scatterplot(data_set_df[0],data_set_df[1])
plt.savefig('display.png') 

def generator(placeholder,h_l_size=[16, 16],reuse=False):
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(placeholder,h_l_size[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,h_l_size[1],activation=tf.nn.leaky_relu)
        out_l = tf.layers.dense(h2,2)

    return out_l
