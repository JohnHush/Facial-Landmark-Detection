import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.learning import train_step
import fetchData
import os
import numpy as np

ckpt_path = '/Users/pitaloveu/working_data/resnet18_tf_checkpoint_from_lilei/try_save/self_save'
#ckpt_path = '/home/jh/working_data/resnet18_face_lilei/try_save/self_save'

data_path = "/Users/pitaloveu/working_data/MTFL"
#data_path = "/home/jh/working_data/MTFL"

def pixel_deviation( x1 , y1 , x2 , y2 ):
    """
    calculate the mean deviation of two pts with ( x1, y1 ) and ( x2, y2)
    are the coordinates of them, respectively.

    the deviation is represented with Eucleadian distance of pts

    the values of the arguments will be in the range of [0 , 1]
    """
    x1 = tf.to_float( x1 )
    y1 = tf.to_float( y1 )
    x2 = tf.to_float( x2 )
    y2 = tf.to_float( y2 )

    error = tf.sqrt( tf.square( x1 - x2 ) * 96 * 96 + tf.square( y1 - y2 ) * 112 * 112 )

    return tf.reduce_mean( error )

def left_eye_deviation( landmark_label , landmark_predict ):
    x1 = landmark_label[ : , 0 ]
    y1 = landmark_label[ : , 5 ]

    x2 = landmark_predict[ : , 0 ]
    y2 = landmark_predict[ : , 5 ]

    return pixel_deviation( x1 , y1 , x2 , y2 )

def right_eye_deviation( landmark_label , landmark_predict ):
    x1 = landmark_label[ : , 1 ]
    y1 = landmark_label[ : , 6 ]

    x2 = landmark_predict[ : , 1 ]
    y2 = landmark_predict[ : , 6 ]

    return pixel_deviation( x1 , y1 , x2 , y2 )

def nose_deviation( landmark_label , landmark_predict ):
    x1 = landmark_label[ : , 2 ]
    y1 = landmark_label[ : , 7 ]

    x2 = landmark_predict[ : , 2 ]
    y2 = landmark_predict[ : , 7 ]

    return pixel_deviation( x1 , y1 , x2 , y2 )

def left_mouth_deviation( landmark_label , landmark_predict ):
    x1 = landmark_label[ : , 3 ]
    y1 = landmark_label[ : , 8 ]

    x2 = landmark_predict[ : , 3 ]
    y2 = landmark_predict[ : , 8 ]

    return pixel_deviation( x1 , y1 , x2 , y2 )

def right_mouth_deviation( landmark_label , landmark_predict ):
    x1 = landmark_label[ : , 4 ]
    y1 = landmark_label[ : , 9 ]

    x2 = landmark_predict[ : , 4 ]
    y2 = landmark_predict[ : , 9 ]

    return pixel_deviation( x1 , y1 , x2 , y2 )

def prelu(_x , variable_scope = None ):
    assert variable_scope is not None

    with tf.variable_scope( variable_scope ):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
    # i suppose tf.Session is more intuitive
    sess = tf.Session()
  
    iterator_test = fetchData.evaluate_input_fn( data_path, 2995 ).make_one_shot_iterator()
    iterator_train = fetchData.train_input_fn( data_path , 128 ).make_one_shot_iterator()

    test_data_ops = iterator_test.get_next()
    train_data_ops = iterator_train.get_next()
    saver = tf.train.import_meta_graph( ckpt_path + '.meta')

    features_test , labels_test = sess.run( test_data_ops )

    graph = tf.get_default_graph()

    input = graph.get_tensor_by_name( 'data:0' )
    label_placeholder = tf.placeholder( tf.float32 , [None, 10] )

    feature = graph.get_tensor_by_name( 'dense/BiasAdd:0' )

    landmark = slim.fully_connected( feature , 10 , \
            activation_fn = tf.sigmoid , scope = 'landmark' )

    loss_landmark = tf.losses.mean_squared_error( \
            label_placeholder , landmark )
    tf.summary.scalar( "loss_landmark" , loss_landmark )

    left_IDev = left_eye_deviation( label_placeholder , landmark )
    right_IDev = right_eye_deviation( label_placeholder , landmark )
    nose_Dev = nose_deviation( label_placeholder , landmark )
    left_MDev = left_mouth_deviation( label_placeholder , landmark )
    right_MDev = right_mouth_deviation( label_placeholder , landmark )

    tf.summary.scalar( "left_I_Dev" , left_IDev )
    tf.summary.scalar( "right_I_Dev" , right_IDev )
    tf.summary.scalar( "nose_Dev" , nose_Dev )
    tf.summary.scalar( "left_Mouth_Dev" , left_MDev )
    tf.summary.scalar( "right_Mouth_Dev" , right_MDev )

    train_op = tf.train.AdamOptimizer( learning_rate = \
            0.0001 ).minimize( loss_landmark )

    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter( './tflog/train', sess.graph)
    test_writer = tf.summary.FileWriter( './tflog/test' )

    sess.run( tf.global_variables_initializer() )
    for i in range( 2000 ):
        features_train, labels_train = sess.run( train_data_ops ) 
        _ , loss , summary = sess.run(\
                [ train_op , loss_landmark , \
                merged ],\
                feed_dict = { input: features_train['image'] , \
                label_placeholder : labels_train['landmarks'] }
                )

        train_writer.add_summary( summary , i )

        if i% 10 == 0:
            loss, summary = sess.run( \
                    [ loss_landmark , merged ], \
                    feed_dict = { input : features_test['image'], \
                    label_placeholder : labels_test['landmarks'] }
                    )
            test_writer.add_summary( summary , i )
            print( "training loss equals to : %f" % loss  )

    train_writer.close()
    test_writer.close()

"""
    graph = tf.Graph()
    with graph.as_default():
        iterator_train = fetchData.train_input_fn( data_path , 128 ).make_one_shot_iterator()
        features_train_batch , labels_train_batch = iterator_train.get_next()
        saver = tf.train.import_meta_graph( ckpt_path + '.meta')
        with tf.Session() as sess:
            writer = tf.summary.FileWriter( './new_graph_input' )

            feature = graph.get_tensor_by_name( 'dense/BiasAdd:0' )

            # input is a placeholder for image
            input = graph.get_tensor_by_name( 'data:0' )
            label_placeholder = tf.placeholder( tf.float32 , [None, 10] )

            # build the header
            landmark = slim.fully_connected( feature , 10 , scope = "landmark" , activation_fn = tf.sigmoid )
            loss_landmark = tf.losses.mean_squared_error( label_placeholder , landmark )

            tf.summary.scalar( 'loss_landmark' , loss_landmark )


            trainable_layers = [ 'landmark' ]
            trainable_list = [ v for v in tf.trainable_variables() if v.name.split('/')[0] \
                    in trainable_layers ]

            train_op = tf.train.AdamOptimizer( learning_rate= 0.0001 ).minimize( loss_landmark , \
                    var_list = trainable_list )

            writer.add_graph( sess.graph )
            
            
            with tf.name_scope( "test_phase" ):
                landmark_test = slim.fully_connected( feature , 10 , scope = "landmark" , activation_fn = tf.sigmoid )
                loss_landmark_test = tf.losses.mean_squared_error( labels_test['landmarks'] , landmark_test )
                tf.summary.scalar( 'loss_landmark_test' , loss_landmark_test )
            

            sess.run( tf.global_variables_initializer() )
            saver.restore( sess , ckpt_path )
            merged_summary_op = tf.summary.merge_all()
            for i in range(20000):
                features_train, labels_train = sess.run( [features_train_batch, labels_train_batch ] )
                _ , loss , summary = sess.run( [ train_op , loss_landmark , merged_summary_op ] , \
                        feed_dict = { input: features_train['image'] , \
                        label_placeholder : labels_train['landmarks'] } )

                writer.add_summary( summary , i )

                if i% 10 == 0:
                    print( "training loss equals to : %f" % loss  )

"""
