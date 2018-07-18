from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow as tf

graph = tf.Graph()

with graph.as_default():

    #saver = tf.train.import_meta_graph( "./slim_mnist_logdir/model.ckpt-4501.meta" )
    #saver = tf.train.import_meta_graph( "/Users/pitaloveu/working_data/resnet18_tf_checkpoint_from_lilei/try_save/self_save.meta" )

    #input_data = graph.get_tensor_by_name( "shuffle_batch:0" )
    s= graph.get_operations()

    my_string = print (s)

    with tf.Session() as sess:
        #saver.restore(sess, tf.train.latest_checkpoint('/Users/pitaloveu/working_data/resnet18_tf_checkpoint_from_lilei/try_save') )

        train_writer = tf.summary.FileWriter( "tb_logdir" )
        train_writer.add_graph(sess.graph)

    #print_tensors_in_checkpoint_file( file_name= "/Users/pitaloveu/working_data/resnet18_tf_checkpoint_from_lilei/try_save/self_save", tensor_name='', all_tensors=False)
    print_tensors_in_checkpoint_file( file_name= "./slim_mnist_logdir/model.ckpt-4501", tensor_name='', all_tensors=False)
