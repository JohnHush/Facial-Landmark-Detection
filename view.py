from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow as tf

graph = tf.Graph()

with graph.as_default():

    saver = tf.train.import_meta_graph( "./slim_mnist_logdir/model.ckpt-4477.meta" )

    #input_data = graph.get_tensor_by_name( "shuffle_batch:0" )
    print( graph.get_operations() )

    #sess = tf.Session()
    #tf.train.write_graph(sess.graph_def, './tmp/my-model', 'train.pbtxt')





#print_tensors_in_checkpoint_file(file_name='./slim_mnist_logdir/model.ckpt-500', tensor_name='', all_tensors=False)
