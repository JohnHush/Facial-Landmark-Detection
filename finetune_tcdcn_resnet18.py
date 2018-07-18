import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.learning import train_step
import fetchData

ckpt_path = '/Users/pitaloveu/working_data/resnet18_tf_checkpoint_from_lilei/try_save/self_save'
data_path = "/Users/pitaloveu/working_data/MTFL"

def prelu(_x , variable_scope = None ):
    assert variable_scope is not None

    with tf.variable_scope( variable_scope ):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

def resnet18( input ):
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

        # add fc layer

        net = slim.flatten( net , scope = "flatten" )
        net = slim.fully_connected( net , 512 , activation_fn = None , scope = "feature" )

    return  net


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()

    with graph.as_default():
        #v1 = tf.Variable( tf.zeros([64]) , name = "test_v1" )
        #v2 = tf.Variable( tf.zeros([3,3,3,64]) , name = "test_v2")

        iterator_train = fetchData.train_input_fn( data_path , \
                batch_size = 64 ).make_one_shot_iterator()

        features , labels = iterator_train.get_next()
        
        # build the whole graph until train_op
        net = resnet18( features['image'] )
        with tf.name_scope( "head" ):
            gender = slim.fully_connected( net , 2 , scope = 'gender' )
            label_gender_oh = tf.one_hot( labels['gender'] - 1, depth = 2 , axis = -1 )
            loss_gender = tf.losses.softmax_cross_entropy( label_gender_oh , gender )

            optimizer = tf.train.AdamOptimizer( learning_rate= 0.0001 )

            train_op = slim.learning.create_train_op( loss_gender, optimizer )
            logdir = "./finetune_shit"

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
        ## define dict to map from weights to weight
        variables_to_restore = dict( zip( old_name , list(map( slim.get_unique_variable , new_name ) ) ) )

        #restorer = tf.train.Saver( variables_to_restore )

        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                ckpt_path , variables_to_restore )

        def InitAssignFn(sess):
            sess.run(init_assign_op, init_feed_dict)

        #with tf.Session() as sess:
        #    sess.run( tf.global_variables_initializer() )
        #    restorer.restore( sess , ckpt_path )
        #    sess.run( net )
        #    print( net.eval() )

        slim.learning.train(
                train_op,
                logdir,
                number_of_steps = 1000,
                graph = graph,
                init_fn = InitAssignFn )

    """
    graph = tf.Graph()

    with graph.as_default():
        # import to default graph
        saver = tf.train.import_meta_graph( \
                "/Users/pitaloveu/working_data/resnet18_tf_checkpoint_from_lilei/try_save/self_save.meta" )

        checkpoint_path = \
                "/Users/pitaloveu/working_data/resnet18_tf_checkpoint_from_lilei/try_save/self_save"

#        conv1 = graph.get_tensor_by_name( "convolution:0" )
#        print ( conv1.shape )

        #variables_to_restore = slim.get_model_variables()

        
        variables_to_restore = { "conv1_1_bias": slim.get_unique_variable("conv1_ft/biases" )}

        print( variables_to_restore )
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                checkpoint_path, variables_to_restore )

        def InitAssignFn(sess):
            sess.run(init_assign_op, init_feed_dict)
        
       
        # load training data using train_input_fn
        iterator_train = fetchData.train_input_fn( "/Users/pitaloveu/working_data/MTFL" , batch_size = 256 ).make_one_shot_iterator()

        features , labels = iterator_train.get_next()

        train_images = features['image']
        label_gender_oh = tf.one_hot( labels['gender'] -1 , depth = 2 , axis = -1 )

        # get input data tensor in the loaded graph
        #input_data_tensor = graph.get_tensor_by_name( "data:0" )
        #input_data_tensor = train_images

        # get feature tensor in the loaded graph
        feature_tensor = graph.get_tensor_by_name( "dense/BiasAdd:0" )

        # get convolutional layer1 tensor in the loaded graph
        # it's shape is [ ? , 56 , 48 , 3 ]
        conv1_tensor = graph.get_tensor_by_name( "convolution:0" )


        # adding data head to walk around the PlaceHolder problem
        #with tf.name_scope( "data_head" ) as scope:
        conv1_tensor = slim.conv2d( train_images , 64 , [3,3] , stride = 2 , scope = "conv1_ft")

        with tf.name_scope( "finetuning" ) as scope:
            with slim.arg_scope( [slim.fully_connected] ,
                    weights_initializer = tf.truncated_normal_initializer( stddev=0.1 ),
                    activation_fn = None ):
                fc_gender = slim.fully_connected( feature_tensor , 2 , scope = "fc_gender" )
                loss_gender = tf.losses.softmax_cross_entropy( label_gender_oh , fc_gender )

                optimizer = tf.train.AdamOptimizer( learning_rate = 0.005 )
                logdir = './slim_tcdcn'
                train_op = slim.learning.create_train_op( loss_gender , optimizer )

                #variables_to_restore = { "conv1_1_bias:0": slim.get_unique_variable("conv1_ft/biases" )}
    #            variables_to_restore = slim.get_model_variables()
                #variables_to_restore =slim.get_variables_to_restore( include=["conv1_2_bias", "conv1_2_weight"] , exclude = ["conv1_ft/biases", "fc_gender", "global_step"])

                #print( variables_to_restore )
                #init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                #        checkpoint_path, variables_to_restore )

                def InitAssignFn(sess):
                    sess.run(init_assign_op, init_feed_dict)

        slim.learning.train( train_op , 
                logdir, 
                number_of_steps = 1000 , 
                save_interval_secs = 100,
                graph = graph)
#                init_fn = InitAssignFn )
    
        """ 
