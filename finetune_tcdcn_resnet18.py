import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.learning import train_step
import fetchData

ckpt_path = '/Users/pitaloveu/working_data/resnet18_tf_checkpoint_from_lilei/try_save/self_save'
ckpt_path = '/home/jh/working_data/resnet18_face_lilei/try_save/self_save'

data_path = "/Users/pitaloveu/working_data/MTFL"
data_path = "/home/jh/working_data/MTFL"

def left_eye_accuracy( landmark_label , landmark_predict ):
    landmark_label = tf.to_float( landmark_label )
    landmark_predict = tf.to_float( landmark_predict )
    left_eye_label_x = landmark_label[ : , 0 ]
    left_eye_label_y = landmark_label[ : , 5 ]

    left_eye_predi_x = landmark_predict[ : , 0 ]
    left_eye_predi_y = landmark_predict[ : , 5 ]

    error = tf.sqrt( tf.square( left_eye_label_x - left_eye_predi_x ) * 96. * 96. + \
            tf.square( left_eye_label_y - left_eye_predi_y ) * 112. * 112. )

    return tf.reduce_mean( error )

def my_accuracy( labels, \
        predictions , \
        weights = None, \
        metrics_collections=None, \
        updates_collections=None, \
        name = None ):
    """
    this metric is specially for landmarks in face detection
    """
    from tensorflow.python.ops import math_ops
    
    absolute_errors = math_ops.abs( predictions - labels )
    mean_t , update_op = tf.metrics.mean( absolute_errors , metrics_collections = \
            metrics_collections , updates_collections = updates_collections , \
            name = "my_accuracy" )

    return mean_t , update_op
    

def prelu(_x , variable_scope = None ):
    assert variable_scope is not None

    with tf.variable_scope( variable_scope ):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

def resnet18( input , is_training ):
    """
    directly implement the network from Lilei's resnet arch
    """
    with slim.arg_scope( [slim.conv2d] , \
            weights_initializer = tf.constant_initializer(0.0) ):
        
        # block 1
        net = slim.conv2d( input , 64 , [3,3] , stride = 2 , scope = "conv1_1" )
        net_copy = net
        net = prelu( net , "prelu1_1" )
        net = slim.conv2d( net   , 64 , [3,3] , stride = 1 , scope = "conv1_2" )
        net = prelu( net , "prelu1_2" )
        net = slim.conv2d( net   , 64 , [3,3] , stride = 1 , scope = "conv1_3" )
        net = prelu( net , 'prelu1_3' )
        net = net + net_copy

        # block 2-1
        net = slim.conv2d( net , 128 , [3,3] , stride = 2 , scope = "conv2_1" )
        net_copy = net
        net = prelu( net , "prelu2_1" )
        net = slim.conv2d( net , 128 , [3,3] , stride = 1 , scope = "conv2_2" )
        net = prelu( net , "prelu2_2" )
        net = slim.conv2d( net , 128 , [3,3] , stride = 1 , scope = "conv2_3" )
        net = prelu( net , 'prelu2_3' )
        net = net + net_copy
        net_copy = net

        # block 2-2
        net = slim.conv2d( net , 128 , [3,3] , stride = 1 , scope = "conv2_4" )
        net = prelu( net , "prelu2_4" )
        net = slim.conv2d( net , 128 , [3,3] , stride = 1 , scope = "conv2_5" )
        net = prelu( net , 'prelu2_5' )
        net = net + net_copy

        # block 3-1
        net = slim.conv2d( net , 256 , [3,3] , stride = 2 , scope = "conv3_1" )
        net_copy = net
        net = prelu( net , "prelu3_1" )
        net = slim.conv2d( net , 256 , [3,3] , stride = 1 , scope = "conv3_2" )
        net = prelu( net , "prelu3_2" )
        net = slim.conv2d( net , 256 , [3,3] , stride = 1 , scope = "conv3_3" )
        net = prelu( net , 'prelu3_3' )
        net = net + net_copy
        net_copy = net

        # block 3-2
        net = slim.conv2d( net , 256 , [3,3] , stride = 1 , scope = "conv3_4" )
        net = prelu( net , "prelu3_4" )
        net = slim.conv2d( net , 256 , [3,3] , stride = 1 , scope = "conv3_5" )
        net = prelu( net , 'prelu3_5' )
        net = net + net_copy
        net_copy = net

        # block 3-3
        net = slim.conv2d( net , 256 , [3,3] , stride = 1 , scope = "conv3_6" )
        net = prelu( net , "prelu3_6" )
        net = slim.conv2d( net , 256 , [3,3] , stride = 1 , scope = "conv3_7" )
        net = prelu( net , 'prelu3_7' )
        net = net + net_copy
        net_copy = net

        # block 3-4
        net = slim.conv2d( net , 256 , [3,3] , stride = 1 , scope = "conv3_8" )
        net = prelu( net , "prelu3_8" )
        net = slim.conv2d( net , 256 , [3,3] , stride = 1 , scope = "conv3_9" )
        net = prelu( net , 'prelu3_9' )
        net = net + net_copy

        # block 4-1
        net = slim.conv2d( net , 512 , [3,3] , stride = 2 , scope = "conv4_1" )
        net_copy = net
        net = prelu( net , "prelu4_1" )
        net = slim.conv2d( net , 512 , [3,3] , stride = 1 , scope = "conv4_2" )
        net = prelu( net , "prelu4_2" )
        net = slim.conv2d( net , 512 , [3,3] , stride = 1 , scope = "conv4_3" )
        net = prelu( net , 'prelu4_3' )
        net = net + net_copy

        regularizer = slim.l2_regularizer( 0. )
        # add fc layer
        net = slim.flatten( net , scope = "flatten" )
        net = slim.fully_connected( net , 512 , activation_fn = None , scope = "feature" )

        net = tf.layers.dropout( net , rate = 0.01 , training = is_training, \
                name = "dropout" )
        # add head for landmark predicting
        # recently ignore all auxiliary characters predicting
        landmark = slim.fully_connected( net , 10 , scope = "landmark" , \
                activation_fn = tf.sigmoid , weights_regularizer = regularizer )

        # test

        pose = slim.fully_connected( net , 5 , scope = "pose" )

    return  landmark , pose

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

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
            landmark , _ = resnet18( features['image'] , True )
            scope.reuse_variables()
            landmark_test , pose_test = resnet18( features_test['image'] , False)

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
            total_loss = slim.losses.get_total_loss()

            optimizer = tf.train.AdamOptimizer( learning_rate= 0.00005 )

            train_op = slim.learning.create_train_op( total_loss, optimizer ,\
                    variables_to_train = trainable_list,\
                    gradient_multipliers = gradient_multipliers )

            logdir = "./resnet18_finetune1"

            loss_landmark_test = tf.losses.mean_squared_error( labels_test['landmarks'] ,\
                    landmark_test )

            #accuracy_test = my_accuracy( labels_test['landmarks'][: , 0 ], 
            #        landmark_test[:,0] )
            accuracy_test = left_eye_accuracy( labels['landmarks'] , landmark )

            # add summaries
            tf.summary.scalar( "loss_landmark" , loss_landmark )
            tf.summary.scalar( "loss_landmark_test" , loss_landmark_test )

        old_name = ['conv1_1_weight', 'conv1_1_bias', 'relu1_1_gamma',\
                    'conv1_2_weight', 'conv1_2_bias', 'relu1_2_gamma',\
                    'conv1_3_weight', 'conv1_3_bias', 'relu1_3_gamma',\

                    'conv2_1_weight', 'conv2_1_bias', 'relu2_1_gamma',\
                    'conv2_2_weight', 'conv2_2_bias', 'relu2_2_gamma',\
                    'conv2_3_weight', 'conv2_3_bias', 'relu2_3_gamma',\
                    'conv2_4_weight', 'conv2_4_bias', 'relu2_4_gamma',\
                    'conv2_5_weight', 'conv2_5_bias', 'relu2_5_gamma',\

                    'conv3_1_weight', 'conv3_1_bias', 'relu3_1_gamma',\
                    'conv3_2_weight', 'conv3_2_bias', 'relu3_2_gamma',\
                    'conv3_3_weight', 'conv3_3_bias', 'relu3_3_gamma',\
                    'conv3_4_weight', 'conv3_4_bias', 'relu3_4_gamma',\
                    'conv3_5_weight', 'conv3_5_bias', 'relu3_5_gamma',\
                    'conv3_6_weight', 'conv3_6_bias', 'relu3_6_gamma',\
                    'conv3_7_weight', 'conv3_7_bias', 'relu3_7_gamma',\
                    'conv3_8_weight', 'conv3_8_bias', 'relu3_8_gamma',\
                    'conv3_9_weight', 'conv3_9_bias', 'relu3_9_gamma',\

                    'conv4_1_weight', 'conv4_1_bias', 'relu4_1_gamma',\
                    'conv4_2_weight', 'conv4_2_bias', 'relu4_2_gamma',\
                    'conv4_3_weight', 'conv4_3_bias', 'relu4_3_gamma',\
                    'dense/kernel', 'dense/bias']

        """
        new_name = ['conv1_1/weights:0', 'conv1_1/biases:0', 'prelu1_1/alpha:0',\
                    'conv1_2/weights:0', 'conv1_2/biases:0', 'prelu1_2/alpha:0',\
                    'conv1_3/weights:0', 'conv1_3/biases:0', 'prelu1_3/alpha:0',\

                    'conv2_1/weights:0', 'conv2_1/biases:0', 'prelu2_1/alpha:0',\
                    'conv2_2/weights:0', 'conv2_2/biases:0', 'prelu2_2/alpha:0',\
                    'conv2_3/weights:0', 'conv2_3/biases:0', 'prelu2_3/alpha:0',\
                    'conv2_4/weights:0', 'conv2_4/biases:0', 'prelu2_4/alpha:0',\
                    'conv2_5/weights:0', 'conv2_5/biases:0', 'prelu2_5/alpha:0',\

                    'conv3_1/weights:0', 'conv3_1/biases:0', 'prelu3_1/alpha:0',\
                    'conv3_2/weights:0', 'conv3_2/biases:0', 'prelu3_2/alpha:0',\
                    'conv3_3/weights:0', 'conv3_3/biases:0', 'prelu3_3/alpha:0',\
                    'conv3_4/weights:0', 'conv3_4/biases:0', 'prelu3_4/alpha:0',\
                    'conv3_5/weights:0', 'conv3_5/biases:0', 'prelu3_5/alpha:0',\
                    'conv3_6/weights:0', 'conv3_6/biases:0', 'prelu3_6/alpha:0',\
                    'conv3_7/weights:0', 'conv3_7/biases:0', 'prelu3_7/alpha:0',\
                    'conv3_8/weights:0', 'conv3_8/biases:0', 'prelu3_8/alpha:0',\
                    'conv3_9/weights:0', 'conv3_9/biases:0', 'prelu3_9/alpha:0',\

                    'conv4_1/weights:0', 'conv4_1/biases:0', 'prelu4_1/alpha:0',\
                    'conv4_2/weights:0', 'conv4_2/biases:0', 'prelu4_2/alpha:0',\
                    'conv4_3/weights:0', 'conv4_3/biases:0', 'prelu4_3/alpha:0',\
                    'feature/weights:0', 'feature/biases:0']
        """
        new_name = ['conv1_1/weights', 'conv1_1/biases', 'prelu1_1/alpha',\
                    'conv1_2/weights', 'conv1_2/biases', 'prelu1_2/alpha',\
                    'conv1_3/weights', 'conv1_3/biases', 'prelu1_3/alpha',\

                    'conv2_1/weights', 'conv2_1/biases', 'prelu2_1/alpha',\
                    'conv2_2/weights', 'conv2_2/biases', 'prelu2_2/alpha',\
                    'conv2_3/weights', 'conv2_3/biases', 'prelu2_3/alpha',\
                    'conv2_4/weights', 'conv2_4/biases', 'prelu2_4/alpha',\
                    'conv2_5/weights', 'conv2_5/biases', 'prelu2_5/alpha',\

                    'conv3_1/weights', 'conv3_1/biases', 'prelu3_1/alpha',\
                    'conv3_2/weights', 'conv3_2/biases', 'prelu3_2/alpha',\
                    'conv3_3/weights', 'conv3_3/biases', 'prelu3_3/alpha',\
                    'conv3_4/weights', 'conv3_4/biases', 'prelu3_4/alpha',\
                    'conv3_5/weights', 'conv3_5/biases', 'prelu3_5/alpha',\
                    'conv3_6/weights', 'conv3_6/biases', 'prelu3_6/alpha',\
                    'conv3_7/weights', 'conv3_7/biases', 'prelu3_7/alpha',\
                    'conv3_8/weights', 'conv3_8/biases', 'prelu3_8/alpha',\
                    'conv3_9/weights', 'conv3_9/biases', 'prelu3_9/alpha',\

                    'conv4_1/weights', 'conv4_1/biases', 'prelu4_1/alpha',\
                    'conv4_2/weights', 'conv4_2/biases', 'prelu4_2/alpha',\
                    'conv4_3/weights', 'conv4_3/biases', 'prelu4_3/alpha',\
                    'feature/weights', 'feature/biases']

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
                accuracy = session.run( train_step_fn.accuracy_test )
                print('Step %s - Loss: %.2f Accuracy: %.2f' % (str(train_step_fn.step).rjust(6, '0' ), total_loss, accuracy ))

            train_step_fn.step += 1

            return [total_loss , should_stop ]

        train_step_fn.step = 0
        train_step_fn.accuracy_test = accuracy_test

        slim.learning.train(
                train_op,
                logdir,
                number_of_steps = 20000,
                graph = graph,
                init_fn = InitAssignFn,
                train_step_fn = train_step_fn,
                save_summaries_secs = 10 )

