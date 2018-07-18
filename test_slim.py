import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.learning import train_step
import numpy as np

def slim_model( input , is_training = True ):
    with tf.name_scope( "reshape" ) as scope:
        net = tf.reshape( input , [-1, 28, 28, 1] )

    with slim.arg_scope( [slim.conv2d , slim.fully_connected] , 
            weights_initializer = tf.truncated_normal_initializer( stddev=0.1 ),
            activation_fn = tf.nn.relu ):
        net = slim.conv2d( net , 32 , [5,5] , scope = "conv1" )
        net = slim.max_pool2d( net , [2,2] , scope = "pool1" )
        net = slim.conv2d( net , 64 , [5,5] , scope = "conv2" )
        net = slim.max_pool2d( net , [2,2] , scope = "pool2" )
        net = slim.flatten( net , scope = "flatten" )
        net = slim.fully_connected( net , 1024 , scope = "fc1" )
        net = slim.dropout( net , keep_prob = 0.5 , is_training= is_training, scope = "dr" )
        net = slim.fully_connected( net , 10 , activation_fn = None , scope = "fc2" )

    return net

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()

    with graph.as_default():
        # load training data from mnist module contained in tf
        # then shuffle it into batch for training
        mnist = input_data.read_data_sets( "/tmp/tensorflow/mnist/input_data" )

        whole_data = mnist.train.next_batch( 60000 )
        batch = tf.train.shuffle_batch( whole_data , batch_size = 50 , \
            capacity = 1000 + 3 * 50 , min_after_dequeue = 1000 , \
            enqueue_many = True , name = "sb" )

        batch_test_images = tf.convert_to_tensor( mnist.test.images , np.float32 , name = "bti")
        batch_test_labels = tf.convert_to_tensor( mnist.test.labels , np.int32  , name = "btl")

        tf.add_to_collection( "test_data" , batch_test_images )
        tf.add_to_collection( "test_data" , batch_test_labels )

        images = batch[0]
        labels = tf.to_int32 ( batch[1] )

        with tf.variable_scope("model") as scope:
            predict_labels = slim_model( images , True )
            scope.reuse_variables()
            #predictions_test = slim_model( mnist.test.images , False )
            predictions_test = slim_model( batch_test_images , False )

        cross_entropy = tf.losses.sparse_softmax_cross_entropy( labels , predict_labels )
        optimizer = tf.train.AdamOptimizer( learning_rate= 0.001, \
                                        beta1 = 0.9 , \
                                        beta2 = 0.99 )
        tf.add_to_collection( "ce" , cross_entropy )
        #tf.add_to_collection( "opti" , optimizer )

        tf.summary.scalar( "loss" , cross_entropy )
        train_op = slim.learning.create_train_op( cross_entropy , optimizer  )

        #tf.add_to_collection( "train_op" , train_op )

        logdir = "./slim_mnist_logdir"

        accuracy_test = slim.metrics.accuracy(tf.to_int32(tf.argmax(predictions_test, 1)), \
                batch_test_labels )

        tf.add_to_collection( "accuracy_test" , accuracy_test )


    def train_step_fn(session, *args, **kwargs):
        total_loss, should_stop = train_step(session, *args, **kwargs)

        if train_step_fn.step % 100 == 0:
            accuracy = session.run( train_step_fn.accuracy_test )
            print('Step %s - Loss: %.2f Accuracy: %.2f%%' % (str(train_step_fn.step).rjust(6, '0'), total_loss, accuracy * 100))

    
        train_step_fn.step += 1
        return [total_loss, should_stop]

    train_step_fn.step = 0
    train_step_fn.accuracy_test = accuracy_test

    slim.learning.train( 
            train_op,
            logdir,
            train_step_fn = train_step_fn,
            graph = graph,
            number_of_steps = 10000,
            save_interval_secs = 30
            )

    #slim.learning.train( train_op, logdir, number_of_steps=500 )
    """
    prediction_test = slim_model( mnist.test.images , False )
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
            { 'accuracy': slim.metrics.accuracy(prediction_test, mnist.test.labels)})

    summary_ops = []
    for metric_name, metric_value in names_to_values.iteritems():
        op = tf.summary.scalar(metric_name, metric_value)
        op = tf.Print(op, [metric_value], metric_name)
        summary_ops.append(op)

    slim.get_or_create_global_step()

    slim.evaluation.evaluation_loop(
            'local',
            log_dir,
            log_dir,
            num_evals=1,
            eval_op=names_to_updates.values(),
            eval_interval_secs = 10 ,
            summary_op=tf.summary.merge(summary_ops) )
    """
