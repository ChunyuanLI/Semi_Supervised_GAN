import argparse
import time
import numpy as np

import tensorflow as tf

import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim

import os
GPUID = 3
os.environ['CUDA_VISIBLE_DEVICES']=str(GPUID)
import pdb

import sys
import plotting
import cifar10_data

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--seed_data', default=1)
parser.add_argument('--count', default=400)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--data_dir', type=str, default='./data/cifar-10-python')
args = parser.parse_args()
print(args)

# fixed random seeds
rng_data = np.random.RandomState(args.seed_data)
rng = np.random.RandomState(args.seed)


def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


# load CIFAR-10
trainx, trainy = cifar10_data.load(args.data_dir, subset='train')
trainx = trainx.transpose(0, 2, 3, 1)

trainx_unl = trainx.copy()
trainx_unl2 = trainx.copy()

testx, testy = cifar10_data.load(args.data_dir, subset='test')
testx = testx.transpose(0, 2, 3, 1)

nr_batches_train = int(trainx.shape[0]/args.batch_size)
nr_batches_test = int(testx.shape[0]/args.batch_size)


# lr = args.learning_rate
data_dtype = 'float32'
batch_size = args.batch_size
latent_size = 100
label_size = 10
img_shape=(batch_size, 32,32,3)


# specify generative model
def generator(input_latent):
    # input_latent = Input(batch_shape=noise_dim, dtype=im_dtype)
    with tf.variable_scope('Net_Gen') as scope:
        xx = layers.fully_connected(input_latent, num_outputs=4*4*512, activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = tf.nn.relu(xx)
        xx = tf.reshape(xx, (batch_size, 4,4,512))
        xx = layers.conv2d_transpose(xx, 256, kernel_size=(5,5), stride=(2, 2), padding='SAME', activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = tf.nn.relu(xx)
        xx = layers.conv2d_transpose(xx, 128, kernel_size=(5,5), stride=(2, 2), padding='SAME', activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = tf.nn.relu(xx)    
        xx = layers.conv2d_transpose(xx, 3, kernel_size=(5,5), stride=(2, 2),  padding='SAME', activation_fn=None)
        xx = layers.batch_norm(xx)
        gen_dat = tf.nn.tanh(xx)

    return gen_dat     

def inference(input_img):
    # input_latent = Input(batch_shape=noise_dim, dtype=im_dtype)
    with tf.variable_scope('Net_Inf') as scope:
        xx = layers.convolution2d(input_img, 128, kernel_size=(5,5), stride=(2, 2), padding='SAME', activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = tf.nn.relu(xx)
        xx = layers.convolution2d(xx, 256, kernel_size=(5,5), stride=(2, 2), padding='SAME', activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = tf.nn.relu(xx)    
        xx = layers.convolution2d(xx, 512, kernel_size=(5,5), stride=(2, 2),  padding='SAME', activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = tf.nn.relu(xx)  
        xx = layers.flatten(xx)
        xx = layers.fully_connected(xx, num_outputs=latent_size, activation_fn=None)
        xx = layers.batch_norm(xx)
        inf_latent = tf.nn.tanh(xx)
    return inf_latent

# specify discriminative model
def discriminator(input_img, input_latent):
    # input_img = Input(batch_shape=(None, 3, 32, 32), dtype=im_dtype)
    with tf.variable_scope('Net_Dis') as scope:
        # pdb.set_trace()
        input_latent_d = tf.tile(tf.reshape(input_latent, [-1,1,1,tf.shape(input_latent)[-1]]), [1,tf.shape(input_img)[1],tf.shape(input_img)[2],1]  )
        xx = tf.concat( [input_latent_d, input_img], axis=3)
        xx = tf.nn.dropout(input_img, 0.8)
        xx = layers.convolution2d(xx, 96, kernel_size=(3,3), stride=(1, 1), padding='same', activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = lrelu(xx)
        xx = layers.convolution2d(xx, 96, kernel_size=(3,3), stride=(1, 1), padding='same', activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = lrelu(xx)
        xx = layers.convolution2d(xx, 96, kernel_size=(3,3), stride=(2, 2), padding='same', activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = lrelu(xx)

        xx = tf.nn.dropout(xx, 0.5)
        xx = layers.convolution2d(xx, 192, kernel_size=(3,3), stride=(1, 1), padding='same', activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = lrelu(xx)
        xx = layers.convolution2d(xx, 192, kernel_size=(3,3), stride=(1, 1), padding='same', activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = lrelu(xx)
        xx = layers.convolution2d(xx, 192, kernel_size=(3,3), stride=(2, 2), padding='same', activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = lrelu(xx)

        xx = tf.nn.dropout(xx, 0.5)
        xx = layers.convolution2d(xx, 192, kernel_size=(3,3), stride=(1, 1), padding='same', activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = lrelu(xx)
        xx = layers.convolution2d(xx, 192, kernel_size=(1,1), stride=(1, 1), padding='same', activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = lrelu(xx)
        xx = layers.convolution2d(xx, 192, kernel_size=(1,1), stride=(1, 1), padding='same', activation_fn=None)
        xx = layers.batch_norm(xx)
        xx = lrelu(xx)
        xx0 = tf.reduce_mean(xx, axis=[1, 2])
        xx = layers.fully_connected(xx0, label_size, activation_fn=None)
        logits = layers.batch_norm(xx)

    return  logits, xx0

# pdb.set_trace()

tf.reset_default_graph()

latent = tf.placeholder(data_dtype, shape=(batch_size, latent_size))
labels = tf.placeholder('int32', shape=(batch_size))
x_lab = tf.placeholder(data_dtype, shape=(batch_size, 32,32,3))
x_unl = tf.placeholder(data_dtype, shape=(batch_size, 32,32,3)) 
lr    = tf.placeholder(data_dtype, shape=()) 

gen_dat = generator(latent)
inf_lat_lab = inference(x_lab)
inf_lat_unl = inference(x_unl)


output_before_softmax_lab, xx_lab = discriminator(x_lab,inf_lat_lab)
output_before_softmax_unl, xx_unl = discriminator(x_unl,inf_lat_unl)
output_before_softmax_gen, xx_gen = discriminator(gen_dat,latent) 


def tf_log_sum_exp(xs):
  maxes = tf.reduce_max(xs, keep_dims=True)
  xs -= maxes
  return tf.squeeze(maxes, [-1]) + tf.log(tf.reduce_sum(tf.exp(xs), -1))

l_lab = tf.gather_nd(output_before_softmax_lab, tf.concat([ tf.expand_dims(tf.range(batch_size), 1), tf.expand_dims(labels,1) ],1))
l_unl = tf_log_sum_exp(output_before_softmax_unl)
l_gen = tf_log_sum_exp(output_before_softmax_gen)

loss_lab_node = -tf.reduce_mean(l_lab) + tf.reduce_mean( tf_log_sum_exp(output_before_softmax_lab))
loss_unl_node = -0.5*tf.reduce_mean(l_unl) + 0.5*tf.reduce_mean(tf.nn.softplus(l_unl)) + 0.5*tf.reduce_mean(tf.nn.softplus(l_gen))

train_err_node = tf.reduce_mean(tf.to_float(tf.not_equal(tf.to_int32(tf.argmax(output_before_softmax_lab, axis=1)),labels)))

# test error
output_before_softmax, xx_lab = discriminator(x_lab, inf_lat_lab)
test_err_node = tf.reduce_mean(tf.to_float(tf.not_equal(tf.to_int32(tf.argmax(output_before_softmax, axis=1)),labels)))


# training the disc net
params_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Net_Dis")
loss_disc = loss_lab_node + args.unlabeled_weight*loss_unl_node

disc_param_updates = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
train_disc_op = disc_param_updates.minimize(loss_disc, var_list=params_disc)

# training the gen net
output_unl = xx_unl
output_gen = xx_gen
m1 = tf.reduce_mean(output_unl,axis=0)
m2 = tf.reduce_mean(output_gen,axis=0)
loss_gen = tf.reduce_mean(tf.abs(m1-m2)) # feature matching loss
params_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Net_Gen")
params_inf = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Net_Inf")
gen_param_updates = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
train_gen_op = gen_param_updates.minimize(loss_gen, var_list=params_gen+params_inf)


# total_parameters = 0
# for variable in tf.trainable_variables():
#     # shape is an array of tf.Dimension
#     shape = variable.get_shape()
#     print(shape)
#     print(len(shape))
#     variable_parametes = 1
#     for dim in shape:
#         print(dim)
#         variable_parametes *= dim.value
#     print(variable_parametes)
#     total_parameters += variable_parametes
# print("Total Parameters: " + str(total_parameters))



# setup session
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.95
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# select labeled data
inds = rng_data.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for i in range(10):
    txs.append(trainx[trainy==i][:args.count])
    tys.append(trainy[trainy==i][:args.count])
txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)

# //////////// perform training //////////////
for epoch in range(1200):
    begin = time.time()
    lr_ = args.learning_rate * np.minimum(3. - epoch/400., 1.)

    # construct randomly permuted minibatches
    trainx = []
    trainy = []
    for t in range(int(np.ceil(trainx_unl.shape[0]/float(txs.shape[0])))):
        inds = rng.permutation(txs.shape[0])
        trainx.append(txs[inds])
        trainy.append(tys[inds])
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]
    
    if epoch==0:
        print(trainx.shape)
        # init_param(trainx[:500]) # data based initialization

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.
    for t in range(nr_batches_train):
        ran_from = t*batch_size
        ran_to = (t+1)*batch_size

        noise_dim = (batch_size, latent_size)
        noise = np.random.uniform(-1,1,size=noise_dim)
        
        ll, lu, te, _ = sess.run([loss_lab_node, loss_unl_node, train_err_node,train_disc_op] , feed_dict={x_lab: trainx[ran_from:ran_to], labels:trainy[ran_from:ran_to], x_unl:trainx_unl[ran_from:ran_to], latent:noise, lr:lr_})

        loss_lab += ll
        loss_unl += lu
        train_err += te

        sess.run([ loss_gen, train_gen_op], feed_dict={ x_unl: trainx_unl2[t*batch_size:(t+1)*batch_size], latent:noise, lr:lr_ })

    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    train_err /= nr_batches_train

    
    # test
    test_err = 0.
    for t in range(nr_batches_test):
        test_err += sess.run(test_err_node , feed_dict={x_lab:testx[t*batch_size:(t+1)*batch_size],labels:testy[t*batch_size:(t+1)*batch_size]})
    test_err /= nr_batches_test

    # report
    print("Iteration %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, train err = %.4f, test err = %.4f" % (epoch, time.time()-begin, loss_lab, loss_unl, train_err, test_err))
    sys.stdout.flush()

    # pdb.set_trace()
    # generate samples from the model
    sample_x = sess.run(gen_dat, feed_dict={latent:noise })
    img_bhwc = sample_x # np.transpose(sample_x[:100,], (0, 2, 3, 1))
    img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title='CIFAR10 samples')
    plotting.plt.savefig("cifar_sample_feature_match.png")

    # save params
    #np.savez('disc_params.npz', *[p.get_value() for p in disc_params])
    #np.savez('gen_params.npz', *[p.get_value() for p in gen_params])
