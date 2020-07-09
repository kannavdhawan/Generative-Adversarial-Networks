import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb 
import tensorflow as tf
sb.set()

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
        h3=tf.layers.dense(h2,2)        
        out_l=tf.layers.dense(h3,1)     #O:Logit prediction for x. Feature transformation learned by x. 
        
    return out_l,h3
        
        
# Adversarial Training 

x_placeholder = tf.placeholder(tf.float32,[None,2]) #real samples 
z_placeholder = tf.placeholder(tf.float32,[None,2]) #random noise samples 
# print(x_placeholder)   #Tensor("Placeholder:0", shape=(?, 2), dtype=float32)
generated_data_out_l=generator(z_placeholder,h_l_size=[16, 16],reuse=False) #feeding random noise 
real_logits,real_rep=discriminator(x_placeholder)#feeding real samples
fake_logits,generated_rep=discriminator(generated_data_out_l,reuse=True)

#Defining loss functions for discrete classification 
"""
Measures the probability error in discrete classification tasks in which each class is independent and not mutually exclusive.
For instance, one could perform multilabel classification where a picture can contain both an elephant and a dog at the same time.

If axis for reduce_mean is None, all dimensions are reduced, and a tensor with a single element is returned.
"""
discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,labels=tf.ones_like(real_logits))+ tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,labels=tf.zeros_like(fake_logits)))
generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,labels=tf.ones_like(fake_logits)))

"""
using get_collection we are collecting the variables/weights for that newtork using scope.
"""

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(generator_loss,var_list = gen_vars) # Generator Train step
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(discriminator_loss,var_list = disc_vars) # Discriminator Train step

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

# sess = tf.Session(config=config)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

batch_size = 256
nd_steps = 10
ng_steps = 10
import os 
x_plot = sample(num=batch_size,scaling=100)

f = open('loss_logs.csv','w')
f.write('Iteration,Discriminator Loss,Generator Loss\n')

for i in range(10001):
    X_batch = sample(num=batch_size,scaling=100)
    Z_batch = sample_Z(batch_size, 2)

    for _ in range(nd_steps):
        _, dloss = sess.run([disc_step, discriminator_loss], feed_dict={x_placeholder: X_batch, z_placeholder: Z_batch})
    rrep_dstep, grep_dstep = sess.run([real_rep, generated_rep], feed_dict={x_placeholder: X_batch, z_placeholder: Z_batch})

    for _ in range(ng_steps):
        _, gloss = sess.run([gen_step, generator_loss], feed_dict={z_placeholder: Z_batch})

    rrep_gstep, grep_gstep = sess.run([real_rep, generated_rep], feed_dict={x_placeholder: X_batch, z_placeholder: Z_batch})

    print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i,dloss,gloss))
    if i%10 == 0:
        f.write("%d,%f,%f\n"%(i,dloss,gloss))

    if i%1000 == 0:
        plt.figure()
        g_plot = sess.run(generated_data_out_l, feed_dict={z_placeholder: Z_batch})
        xax = plt.scatter(x_plot[:,0], x_plot[:,1])
        gax = plt.scatter(g_plot[:,0],g_plot[:,1])

        plt.legend((xax,gax), ("Real Data","Generated Data"))
        plt.title('Samples at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig(os.path.join('plot/epochs/','iteration_%d.png'%i))
        plt.close()

        plt.figure()
        rrd = plt.scatter(rrep_dstep[:,0], rrep_dstep[:,1], alpha=0.5)
        rrg = plt.scatter(rrep_gstep[:,0], rrep_gstep[:,1], alpha=0.5)
        grd = plt.scatter(grep_dstep[:,0], grep_dstep[:,1], alpha=0.5)
        grg = plt.scatter(grep_gstep[:,0], grep_gstep[:,1], alpha=0.5)


        plt.legend((rrd, rrg, grd, grg), ("Real Data Before G step","Real Data After G step",
                               "Generated Data Before G step","Generated Data After G step"))
        plt.title('Transformed Features at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig(os.path.join('plot/features/','feature_transform_%d.png'%i))
        plt.close()

        plt.figure()

        rrdc = plt.scatter(np.mean(rrep_dstep[:,0]), np.mean(rrep_dstep[:,1]),s=100, alpha=0.5)
        rrgc = plt.scatter(np.mean(rrep_gstep[:,0]), np.mean(rrep_gstep[:,1]),s=100, alpha=0.5)
        grdc = plt.scatter(np.mean(grep_dstep[:,0]), np.mean(grep_dstep[:,1]),s=100, alpha=0.5)
        grgc = plt.scatter(np.mean(grep_gstep[:,0]), np.mean(grep_gstep[:,1]),s=100, alpha=0.5)

        plt.legend((rrdc, rrgc, grdc, grgc), ("Real Data Before G step","Real Data After G step",
                               "Generated Data Before G step","Generated Data After G step"))

        plt.title('Centroid of Transformed Features at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig(os.path.join('plot/features/','feature_transform_centroid_%d.png'%i))
        plt.close()

f.close()
   

# data_set=sample(10000,100) 
# print(data_set)
# # print(data_set.shape)

# data_set_df=pd.DataFrame(data_set)
# # plt.plot(data_set_df[0],data_set_df[1],'o',linewidth=2, markersize=2)
# sns.scatterplot(data_set_df[0],data_set_df[1])
# plt.savefig('display.png') 