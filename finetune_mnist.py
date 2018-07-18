import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.learning import train_step

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()

    with graph.as_default():
        # add to default graph
        saver = tf.train.import_meta_graph( "./slim_mnist_logdir/model.ckpt-1000.meta" )
        #batch_input = graph.get_tensor_by_name( "sb:0" )
        #fc2_out = graph.get_tensor_by_name( "model/fc2/BiasAdd:0" )

        checkpoint_path = "./slim_mnist_logdir2/model.ckpt-200"

        variables_to_restore = slim.get_model_variables()
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                checkpoint_path, variables_to_restore )

        def InitAssignFn(sess):
            sess.run(init_assign_op, init_feed_dict)
        
        # load training data from mnist module contained in tf
        # then shuffle it into batch for training
        mnist = input_data.read_data_sets( "/tmp/tensorflow/mnist/input_data" )

        whole_data = mnist.train.next_batch( 60000 )
        batch_train = graph.get_tensor_by_name( "sb:0" )
        batch_train = \
                tf.train.shuffle_batch( whole_data , batch_size = 50 , \
                capacity = 1000 + 3 * 50 , min_after_dequeue = 1000 , \
                enqueue_many = True )

        tf.get_collection( "test_data" )[0] = mnist.test.images
        tf.get_collection( "test_data" )[1] = mnist.test.labels

        accuracy_test = tf.get_collection( "accuracy_test" )[0]

        #batch_input = batch

        #images = batch[0]
        #labels = tf.to_int32 ( batch_input[1] )

        #with tf.variable_scope( "finetuning" ) as scope:
        #    cross_entropy = tf.losses.sparse_softmax_cross_entropy( labels , fc2_out )

        #    optimizer = tf.train.AdamOptimizer( learning_rate= 0.001, \
        #                                 beta1 = 0.9 , \
        #                                 beta2 = 0.99 )

        #    train_op = slim.learning.create_train_op( cross_entropy , optimizer )
        logdir = "./slim_mnist_logdir2"

        #train_op = tf.get_collection( "train_op" )[0]

        cross_entropy = tf.get_collection( "ce" )[0]

        def train_step_fn(session, *args, **kwargs):
            total_loss, should_stop = train_step(session, *args, **kwargs)

            if train_step_fn.step % 100 == 0:
                accuracy = session.run( train_step_fn.accuracy_test )
                print('Step %s - Loss: %.2f Accuracy: %.2f%%' % (str(train_step_fn.step).rjust(6, '0'), total_loss, accuracy * 100))

    
            train_step_fn.step += 1
            return [total_loss, should_stop]

        with tf.variable_scope( "finetuning" ) as scope:

            optimizer = tf.train.AdamOptimizer( learning_rate= 0.001, \
                                                beta1 = 0.9 , \
                                                beta2 = 0.99 )

            train_op = slim.learning.create_train_op( cross_entropy , optimizer  )


            train_step_fn.step = 0
            train_step_fn.accuracy_test = accuracy_test

            slim.learning.train( 
                    train_op,
                    logdir,
                    train_step_fn = train_step_fn,
                    init_fn = InitAssignFn,
                    graph = graph,
                    number_of_steps = 200)

        """
        with tf.variable_scope("model") as scope:
            predict_labels = slim_model( images , True )
            scope.reuse_variables()
            predictions_test = slim_model( mnist.test.images , False )

        cross_entropy = tf.losses.sparse_softmax_cross_entropy( labels , predict_labels )
        optimizer = tf.train.AdamOptimizer( learning_rate= 0.001, \
                                        beta1 = 0.9 , \
                                        beta2 = 0.99 )

        tf.summary.scalar( "loss" , cross_entropy )
        train_op = slim.learning.create_train_op( cross_entropy , optimizer )
        logdir = "./slim_mnist_logdir"

        accuracy_test = slim.metrics.accuracy(tf.to_int32(tf.argmax(predictions_test, 1)), tf.to_int32( mnist.test.labels ) )


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
            number_of_steps = 10000
            )
    """
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
