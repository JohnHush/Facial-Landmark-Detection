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
   
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph( ckpt_path + '.meta')
        with tf.Session() as sess:
            saver.restore( sess , ckpt_path )
            
            feature = graph.get_tensor_by_name( 'dense/BiasAdd:0' )
            input = graph.get_tensor_by_name( 'data:0' )

            print( feature.shape )
            print( input.shape )

    """
    # before that import all training and testing data as numpy arrays
    train_imgs , train_landmarks , train_gender , train_smile , train_glasses , train_pose = \
            fetchData.fetch_numpy_arrays( data_path , is_train = True )

    graph = tf.Graph()
    iterator_test = fetchData.evaluate_input_fn( data_path, 2995 ).make_one_shot_iterator()

    with tf.Session() as sess:
        features_test , labels_test = sess.run( iterator_test.get_next() )

    with graph.as_default():
        # add data placeholder for training
        training_imgs_placeholder = tf.placeholder( train_imgs.dtype , train_imgs.shape )
        training_dataset = fetchData.train_input_fn_v2( training_imgs_placeholder ,\
                train_landmarks , train_gender , train_smile , train_glasses , \
                train_pose , batch_size = 256 )

        training_init_iterator  = training_dataset.make_initializable_iterator()
        #training_fetch_iterator = training_dataset.make_one_shot_iterator()
        #features , labels = training_fetch_iterator.get_next()
        features , labels = training_init_iterator.get_next()

        init_op = training_init_iterator.initializer
        init_feed_dicts = { training_imgs_placeholder : train_imgs }

        # build the whole graph till train_op
        with tf.variable_scope( "base" ) as scope:
            landmark = resnet18( features['image'] , True )
            scope.reuse_variables()
            landmark_test = resnet18( features_test['image'] , False)

        # specify variables wanna be trained
        trainable_layers = [ 'landmark' , 'feature' ]
        trainable_list = [ v for v in tf.trainable_variables() if v.name.split('/')[1] \
                in trainable_layers]

        gradient_multipliers = { 
                'base/feature/weights' : 0.01,
                'base/feature/biases' : 0.01 }

        #print( trainable_list )

        with tf.name_scope( "head" ):
            loss_landmark = tf.losses.mean_squared_error( labels['landmarks'] , landmark )
            total_loss = loss_landmark
            #total_loss = slim.losses.get_total_loss()

            optimizer = tf.train.AdamOptimizer( learning_rate= 0.0001 )

            train_op = slim.learning.create_train_op( total_loss, optimizer ,\
                    variables_to_train = trainable_list,\
                    gradient_multipliers = gradient_multipliers )

            logdir = "./resnet18_finetune1"

            loss_landmark_test = tf.losses.mean_squared_error( labels_test['landmarks'] ,\
                    landmark_test )

            left_eye_deviation = left_eye_deviation( labels_test['landmarks'] , landmark_test )
            right_eye_deviation = right_eye_deviation( labels_test['landmarks'] , landmark_test )
            nose_deviation = nose_deviation( labels_test['landmarks'] , landmark_test )
            left_mouth_deviation = left_mouth_deviation( labels_test['landmarks'] , landmark_test )
            right_mouth_deviation = right_mouth_deviation( labels_test['landmarks'] , landmark_test )

            # add summaries
            tf.summary.scalar( "loss_landmark" , loss_landmark )
            tf.summary.scalar( "loss_landmark_test" , loss_landmark_test )
            tf.summary.scalar( "left_eye_deviation" , left_eye_deviation )
            tf.summary.scalar( "right_eye_deviation" , right_eye_deviation )
            tf.summary.scalar( "nose_deviation" , nose_deviation )
            tf.summary.scalar( "left_mouth_deviation" , left_mouth_deviation )
            tf.summary.scalar( "right_mouth_deviation" , right_mouth_deviation )

        new_name_base_name = list( map(lambda s: "base/" + s , new_name ) )
        ## define dict to map from weights to weight
        variables_to_restore = dict( zip( old_name , list(map( slim.get_unique_variable , new_name_base_name ) ) ) )

        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                ckpt_path , variables_to_restore )

        def InitAssignFn(sess):
            sess.run(init_op, init_feed_dicts )
            sess.run(init_assign_op, init_feed_dict)

        def train_step_fn( session , *args , **kwargs ):
            total_loss, should_stop = train_step(session, *args, **kwargs)

            if train_step_fn.step % 100 == 0:
                left_eye_deviation = session.run( train_step_fn.led )
                right_eye_deviation = session.run( train_step_fn.red )
                nose_deviation = session.run( train_step_fn.nd )
                left_mouth_deviation = session.run( train_step_fn.lmd )
                right_mouth_deviation = session.run( train_step_fn.rmd )

            train_step_fn.step += 1

            return [total_loss , should_stop ]

        train_step_fn.step = 0
        train_step_fn.led = left_eye_deviation
        train_step_fn.red = right_eye_deviation
        train_step_fn.nd = nose_deviation
        train_step_fn.lmd = left_mouth_deviation
        train_step_fn.rmd = right_mouth_deviation

        slim.learning.train(
                train_op,
                logdir,
                number_of_steps = 20000,
                graph = graph,
                init_fn = InitAssignFn,
                train_step_fn = train_step_fn,
                save_summaries_secs = 10 )
    """

