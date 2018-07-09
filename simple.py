# implement Facial Landmark Detection of TCDCN
# using tensorflow custom Estimator and Dataset API
# written by John Hush in Chengdu
# 06/19/2018
# cft

import tensorflow as tf
from fetchData import *

from tensorflow.python import debug as tf_debug

def model( features, labels, mode, params ):

    # directly use features and labels dict
    images = features['image']

    label_landmark = labels['landmarks']
    label_gender   = labels['gender']  - 1
    label_smile    = labels['smile']   - 1
    label_glasses  = labels['glasses'] - 1
    label_pose     = labels['pose']    - 1 

    # generate one-hot tensor for classification tasks

    label_gender_oh = tf.one_hot( label_gender, depth = 2 , axis = -1 )
    label_smile_oh = tf.one_hot( label_smile , depth = 2 , axis = -1 )
    label_glasses_oh = tf.one_hot( label_glasses, depth = 2 , axis = -1 )
    label_pose_oh = tf.one_hot( label_pose , depth = 5 , axis = -1 )

    # using tf.layers module to build the rest of the layers
    # maybe next time could try Keras ^_^

    conv1 = tf.layers.conv2d(
            inputs = images,
            filters = 16,
            kernel_size = [5,5], 
            activation = tf.nn.relu )

    pool1 = tf.layers.max_pooling2d( inputs=conv1, pool_size=[2, 2], strides=2 )

    conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 48,
            kernel_size = [3,3],
            activation = tf.nn.relu )
    
    pool2 = tf.layers.max_pooling2d( inputs=conv2, pool_size=[2, 2], strides=2 )

    conv3 = tf.layers.conv2d(
            inputs = pool2,
            filters = 64,
            kernel_size = [3,3],
            activation = tf.nn.relu )
    
    pool3 = tf.layers.max_pooling2d( inputs=conv3, pool_size=[2, 2], strides=2 )

    conv4 = tf.layers.conv2d(
            inputs = pool3,
            filters = 64,
            kernel_size = [2,2],
            activation = tf.nn.relu )

    # flatten the tensor
    conv4_flatten = tf.reshape( conv4 , shape = [ -1, 2 * 2 * 64 ] )

    # shared feature
    shared_feature = tf.layers.dense(
            inputs = conv4_flatten , units = 100, activation = tf.nn.relu )

    # drop out 
    shared_feature_dropout = tf.layers.dropout(
            inputs = shared_feature , rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN )

    # adding multiple headers
    landmarks = tf.layers.dense( inputs = shared_feature_dropout , units = 10 )
    gender    = tf.layers.dense( inputs = shared_feature_dropout , units = 2 )
    smile     = tf.layers.dense( inputs = shared_feature_dropout , units = 2 )
    glasses   = tf.layers.dense( inputs = shared_feature_dropout , units = 2 )
    pose      = tf.layers.dense( inputs = shared_feature_dropout , units = 5 )

    predictions = {
            "landmarks" : landmarks ,
            "gender" : tf.argmax( input = gender , axis = 1),
            "gender_probability" : tf.nn.softmax( gender , name = "gender_softmax" ),
            "smile": tf.argmax( input = smile , axis = 1 ),
            "smile_probability" : tf.nn.softmax( smile , name = "smile_softmax" ),
            "glasses": tf.argmax( input = glasses , axis = 1 ),
            "glasses_probability" : tf.nn.softmax( glasses , name = "glasses_softmax" ),
            "psoe": tf.argmax( input = pose , axis = 1 ),
            "pose_probability" : tf.nn.softmax( pose , name = "pose_softmax" )
            }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec( mode=mode, predictions=predictions )

    # generate LOSS
    loss_landmark = tf.losses.mean_squared_error( label_landmark , landmarks )
    loss_gender   = tf.losses.softmax_cross_entropy( label_gender_oh , gender )
    loss_smile    = tf.losses.softmax_cross_entropy( label_smile_oh , smile )
    loss_glasses  = tf.losses.softmax_cross_entropy( label_glasses_oh , glasses )
    loss_pose     = tf.losses.softmax_cross_entropy( label_pose_oh , pose )

    #total_loss = loss_landmark + loss_gender + loss_smile + loss_glasses + loss_pose
    total_loss = loss_gender + loss_smile + loss_glasses + loss_pose

    # specify all the configures of Estimator
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005)
        train_op = optimizer.minimize(
                loss = loss_pose,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec( mode=mode , loss = loss_pose , train_op = train_op )

    evaluate_metric_op = {
            "mean_cosine_distance_landmarks" : tf.metrics.mean_cosine_distance(
                label_landmark , landmarks , dim = 1 ) ,

            "accuracy_gender" : tf.metrics.accuracy(
                label_gender , predictions = predictions['gender'] ),

            "accuracy_smile" : tf.metrics.accuracy(
                label_smile , predictions = predictions['smile'] ),

            "accuracy_glasses" : tf.metrics.accuracy(
                label_glasses , predictions = predictions['glasses'] ),

            "accuracy_pose" : tf.metrics.accuracy(
                label_pose , predictions = predictions['pose'] )
            }

    return tf.estimator.EstimatorSpec(
            mode=mode, loss = total_loss, eval_metric_ops = evaluate_metric_op )

if __name__ == "__main__":
    # make a regressor
    tcdcn_regressor = tf.estimator.Estimator(
            model_fn = model , model_dir="/tmp/tcdcn_test_model")

    tensors_to_log = { "pose_softmax" }
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

    #debug_hook = tf_debug.LocalCLIDebugHook()

    #debug_hook = tf_debug.TensorBoardDebugHook("JohndeMacBook-Pro.local:2333")

    tcdcn_regressor.train( 
            input_fn = lambda : train_eval_input_fn( "/Users/pitaloveu/working_data/MTFL" 
                , if_train = True , batch_size = 128 ) ,
            steps = 10000 , 
            hooks = [logging_hook] )
