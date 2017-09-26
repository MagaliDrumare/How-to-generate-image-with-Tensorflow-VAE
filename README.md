


'''
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

from tensorflow.examples.tutorial.mnist import input_data 
mnist=input.data.read_data_sets ('/tmp/data', one_hot= True)
	)

n_pixels=28*28

#input the images 
x=tf.placeholder (tf/flaot32, shape=[(None, n_pixels)])

# Variables 

def weight_variable(shape, name): 

	initial=tf.truncated_normal(shape, stddev=0.1)

	return tf.Variable(initial, name=name)

def bias_variable(shape, name)	:
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

#full conected layer 
def FC_layer(X, W, b)

return tf.matmul(W,X)+b

#hidden / latent part 
latent_dim= 20 
h_dim = 500 # 500 neunrons


## Encoder tanh + simple FC_layer
#layer 1
w_enc= weight_variable([n_pixels, h_dim], 'W_enc')
b_enc= bias_variable([h_dim], 'b_enc')
#tanh is the activation function 
h_enc = tf.nn.tanh(FC_Layer(X,W_enc,b_enc))

#layer 2-mean
W_mu = weight_variable([h_dim, latent_dim], 'W_mu')
b_mu = bias_variable([latent_dim], 'b_mu')
mu = FC_layer(h_enc, W_mu, b_mu) #mean

#standard deviation 
W_logstd = weight_variable([h_dim, latent_dim], 'W_logstd')
b_logstd = bias_variable([latent_dim], 'b_logstd')
logstd = FC_layer(h_enc, W_logstd, b_logstd) #mean

#we backpropagate with the mean and the standard deviation 

#Randomness 
noise =tf.random_normal ([1,latent_dim])

#z is the ultimate output of our encoder
z=mu+tf.mul(noise, tf.exp(.5*logstd))


## Decoder tanh + sigmoid 

#Layer 1
W_dec = weight_variable([latent_dim, h_dim], 'W_dec')
b_dec = bias_variable([h_dim], 'b_dec')
#pass in z here (and the weights and biases we just defined)
h_dec = tf.nn.tanh(FC_layer(z, W_dec, b_dec))


#Layer 2, using the original n pixels here since thats the dimensiaonlty
#we want to restore our data to
W_reconstruct = weight_variable([h_dim, n_pixels], 'W_reconstruct')
b_reconstruct = bias_variable([n_pixels], 'b_reconstruct')
#784 bernoulli parameters output
reconstruction = tf.nn.sigmoid(FC_layer(h_dec, W_reconstruct, b_reconstruct))


#The loss function 
# Variational lower bound  = Log Likelihood -LK divergence 
# Log Likelihood : make sure the reconstructed image is similar as the input image. (le + grand possible)
# LK divergence (cluster) : make sure the value that we encode are closed in the hidden space. (le + petit possible)
# Objectif est de maximiser Varational lowerbound 
#

# Log Likelihood 
log_likelihood = tf.reduce_sum(X*tf.log(reconstruction + 1e-9)+(1 - X)*tf.log(1 - reconstruction + 1e-9), reduction_indices=1)
# KL divergence 
KL_term = -.5*tf.reduce_sum(1 + 2*logstd - tf.pow(mu,2) - tf.exp(2*logstd), reduction_indices=1)


#but  but because tensorflow doesn't have a 'maximizing' optimizer, we minimize the negative lower bound.
variational_lower_bound = tf.reduce_mean(log_likelihood - KL_term)
optimizer = tf.train.AdadeltaOptimizer().minimize(-variational_lower_bound)

```








'''
