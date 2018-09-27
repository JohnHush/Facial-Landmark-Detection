import tensorflow as tf
import tensorflow.contrib.slim as slim
import fetchData
import os
import numpy as np
from CONFIGURES import args
import tensorflow_hub as hub

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    export_path = "./model"

    anno_path = "/home/public/data/celebrity_lmk"
    data_path = "/home/public/data"

    args.data_path = data_path
    args.anno_path = anno_path
    args.tempdir_path = "/home/public/data/tmp"
    args.img_height = 96
    args.img_width  = 96
    args.train_batch_size = 128
    args.test_batch_size  = 128
    args.if_augmentation = True

    data = fetchData.MSCELEB( args )

    trainData_op = data.trainDataStream()
    testData_op  = data.testDataStream()

    sess = tf.Session()

    m = hub.Module( os.path.join( "/home/jh/working_data/models/tensorflow_hub/"
        "8120b7321d9e14533232b1ddd4a74db35324b638" ) , trainable = True  )

    #tf.saved_model.loader.load( sess , [tf.saved_model.tag_constants.TRAINING] , \
    #        saved_model_dir )

    graph = tf.get_default_graph()

    imgs = graph.get_tensor_by_name( "module/hub_input/images:0" )
    imgs_feature = graph.get_tensor_by_name( \
            "module/hub_output/feature_vector/SpatialSqueeze:0" )

    labels = tf.placeholder( tf.float32 , [None, 10] )
    landmarks = slim.fully_connected( imgs_feature , 10 , scope = "landmark", \
            activation_fn = tf.sigmoid )
    loss = tf.losses.mean_squared_error( labels, landmarks ) 

    tf.summary.scalar( "loss" , loss )

    left_IDev = left_eye_deviation( labels , landmarks )
    right_IDev = right_eye_deviation( labels , landmarks )
    nose_Dev = nose_deviation( labels , landmarks )
    left_MDev = left_mouth_deviation( labels , landmarks )
    right_MDev = right_mouth_deviation( labels , landmarks )

    tf.summary.scalar( "left_I_Dev" , left_IDev )
    tf.summary.scalar( "right_I_Dev" , right_IDev )
    tf.summary.scalar( "nose_Dev" , nose_Dev )
    tf.summary.scalar( "left_Mouth_Dev" , left_MDev )
    tf.summary.scalar( "right_Mouth_Dev" , right_MDev )

    #for v in tf.global_variables():
    #    tf.add_to_collection( 'trainable_variables' , v )

    #trainable_layers = [ 'landmark' ]
    #trainable_list = [ v for v in tf.trainable_variables() if \
    #        v.name.split('/')[0] in trainable_layers]
    #trainable_list = tf.trainable_variables()

    #print( trainable_list )

    trainable_list = tf.trainable_variables()
    train_op = tf.train.AdamOptimizer( learning_rate = 0.0001 ).minimize( loss , \
            var_list = trainable_list )

    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter( './tflog/train', graph = graph )
    test_writer = tf.summary.FileWriter( './tflog/test' )

    sess.run( tf.global_variables_initializer() )

    #tf.saved_model.simple_save( sess , './trained_model' , \
    #          inputs = { 'images' : input } , \
    #          outputs = { 'landmarks' : landmark } )

    saver = tf.train.Saver()

    for i in range( 100000 ):
        train_images , train_landmarks = sess.run( trainData_op )
        _ , LOSS , summary = sess.run( [ train_op , loss , merged ],\
                feed_dict = { imgs: train_images , labels : train_landmarks })

        train_writer.add_summary( summary , i )
        print( "training loss equals to : %f" % LOSS  )

        if i%500 == 0:
            save_path = saver.save( sess , "./tmp/model.ckpt" )
            print( "Model saved in path: %s" % save_path )

        if i%50 == 0:
            test_images , test_landmarks = sess.run( testData_op )
            LOSS, summary = sess.run( [ loss , merged ], \
                    feed_dict = { imgs: test_images, labels: test_landmarks } )

            test_writer.add_summary( summary , i )

    train_writer.close()
    test_writer.close()

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    tensor_info_input  = tf.saved_model.utils.build_tensor_info( imgs )
    tensor_info_output = tf.saved_model.utils.build_tensor_info( landmarks )

    prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs  = { 'images' : tensor_info_input } ,
                outputs = { 'landmarks' : tensor_info_output} ,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images':
                prediction_signature,
                },
            main_op=tf.tables_initializer(),
            strip_default_attrs=True)
    builder.save()

