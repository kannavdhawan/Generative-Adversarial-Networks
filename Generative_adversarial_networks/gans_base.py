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

def generator(z_placeholder,h_l_size,reuse):
    """
    Args:
        Placeholder for random sample Z.
        Hidden_layers size in fully connected network 
        reuse to reuse the same layers.
        
    """
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(z_placeholder,h_l_size[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,h_l_size[1],activation=tf.nn.leaky_relu)
        out_l = tf.layers.dense(h2,2)

    return out_l


def discriminator(x_placeholder,h_l_size=[16,16],reuse=False):
    """
    Takes input placeholder for samples from the v_space of real dataset.{samples can be real or generated.}
    
    
    
    """
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        h1=tf.layers.dense(x_placeholder,h_l_size[0],activation=tf.nn.leaky_relu)
        h2=tf.layers.dense(h1,h_l_size[1],activation=tf.nn.leaky_relu) 
        h3=tf.layers.dense(h2,2)        #O: Logit prediction for x
        out_l=tf.layers.dense(h3,1)     #O: Feature transformation learned by x. 
        
    return out_l,h3
        
        
# Adversarial Training 

x_placeholder = tf.placeholder(tf.float32,[None,2]) #real samples 
z_placeholder = tf.placeholder(tf.float32,[None,2]) #random noise samples 
# print(x_placeholder)   #Tensor("Placeholder:0", shape=(?, 2), dtype=float32)
genrated_data_out_l=generator(z_placeholder,h_l_size=[16, 16],reuse=False) #feeding random noise 
disriminator_logits,


   