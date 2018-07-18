# implement Facial Landmark Detection of TCDCN
# using tensorflow custom Estimator and Dataset API
# written by John Hush in Chengdu
# 06/19/2018
# cft

import tensorflow as tf
import os
import fetchData
#from fetchData import *
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug

def slim_model( features , is_training = True ):
    with slim.arg_scope( [ slim.conv2d , slim.fully_connected ] , 
            activation_fn = tf.nn.relu , 
            weights_initializer = tf.truncated_normal_initializer( 0., 0.1 ) ,
            weights_regularizer = slim.l2_regularizer( 0.0005 ) ):
        images = features['image']
        net = slim.conv2d( images , 16 , [5,5] , scope = "conv1" )
        net = slim.max_pool2d( net , [2,2] , scope = "pool1" )
        net = slim.conv2d( net , 48 , [3,3] , scope = "conv2" )
        net = slim.max_pool2d( net , [2,2] , scope = "pool2" )
        net = slim.conv2d( net , 64 , [3,3] , scope = "conv3" )
        net = slim.max_pool2d( net , [2,2] , scope = "pool3" )
        net = slim.conv2d( net , 64 , [2,2] , scope = "conv4" )
        net = slim.flatten( net , scope = "flatten" )
        net = slim.fully_connected( net , 100 , scope = "fc1" )
        net = slim.dropout( net , keep_prob = 0.5 , is_training = is_training , scope = "dr1" )

        # add head
        with slim.arg_scope( [slim.fully_connected ], activation_fn = None ):
            landmark = slim.fully_connected( net , 10 , scope = "landmark" )
            gender   = slim.fully_connected( net , 2  , scope = "gender" )
            smile    = slim.fully_connected( net , 2  , scope = "smile" )
            glasses  = slim.fully_connected( net , 2  , scope = "glasses" )
            pose     = slim.fully_connected( net , 5  , scope = "pose" )
            landmark = tf.sigmoid( landmark )

        return landmark , gender , smile , glasses , pose

if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.INFO)

    #debug_hook = tf_debug.LocalCLIDebugHook()
    #debug_hook = tf_debug.TensorBoardDebugHook("JohndeMacBook-Pro.local:2333")

    iterator = fetchData.train_input_fn( "/Users/pitaloveu/working_data/MTFL" ,\
            batch_size = 256 ).make_one_shot_iterator()

    features , labels = iterator.get_next()

    landmark , gender , smile , glasses , pose = slim_model( features , True )

    label_landmark = labels['landmarks']
    label_gender   = labels['gender']  - 1
    label_smile    = labels['smile']   - 1
    label_glasses  = labels['glasses'] - 1
    label_pose     = labels['pose']    - 1

    label_gender_oh = tf.one_hot( label_gender, depth = 2 , axis = -1 )
    label_smile_oh = tf.one_hot( label_smile , depth = 2 , axis = -1 )
    label_glasses_oh = tf.one_hot( label_glasses, depth = 2 , axis = -1 )
    label_pose_oh = tf.one_hot( label_pose , depth = 5 , axis = -1 )

    reg_loss = tf.add_n( slim.losses.get_regularization_losses() )
    loss_landmark = tf.losses.mean_squared_error( label_landmark , landmark )
    loss_gender   = tf.losses.softmax_cross_entropy( label_gender_oh , gender )
    loss_smile    = tf.losses.softmax_cross_entropy( label_smile_oh , smile )
    loss_glasses  = tf.losses.softmax_cross_entropy( label_glasses_oh , glasses )
    loss_pose     = tf.losses.softmax_cross_entropy( label_pose_oh , pose )

    total_loss = slim.losses.get_total_loss()

    tf.summary.scalar( "loss_gender" , loss_gender )

    optimizer = tf.train.AdamOptimizer( learning_rate= 0.001, \
                                        beta1 = 0.9 , \
                                        beta2 = 0.99 )

    train_op = slim.learning.create_train_op( loss_gender, optimizer )
    logdir = "./slim_logdir"

    slim.learning.train( train_op, logdir, number_of_steps=1000, \
            save_summaries_secs=300, save_interval_secs=600 )
