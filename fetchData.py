import cv2
import tensorflow as tf
import numpy as np
import os.path as op
import matplotlib.pyplot as plt

data_path = "/Users/pitaloveu/working_data/MTFL"
train_file_path = "/Users/pitaloveu/working_data/MTFL/training.txt"

def load_data_path( data_path , file_path ):

    # in the train file, 
    # formatting as [ "image path", 10 * landmark(float), 4*character(int) ]
    reader_handler = open( file_path , "r" )

    image_path_list, landmark_list, gender, smile, glasses, pose = [], [], [] , [], [], []

    for line in reader_handler:
        line  = line.strip()
        elems = line.split( " " )
        elems[0] = elems[0].replace( "\\" , "/" )
        if not op.isfile( op.join( data_path , elems[0] ) ):
            continue

        image_path_list.append( op.join( data_path , elems[0] ) )
        
        landmark_list.append( [float(elems[1]), float(elems[2]) , 
            float(elems[3]) , float(elems[4]), float(elems[5]) , 
            float(elems[6]), float(elems[7]) , float(elems[8]) , 
            float(elems[9]) , float(elems[10]) ] )

        gender.append( int(elems[11]) )
        smile.append( int(elems[12]) )
        glasses.append( int(elems[13]) )
        pose.append( int(elems[14]) )

    return image_path_list , landmark_list, gender, smile, glasses, pose

def fetch_image( image_path ):
    content = tf.read_file( image_path )
    tf_image = tf.image.decode_jpeg( content , channels = 3 )
    resized_image = tf.image.resize_images( tf_image , [ 224 , 224 ] )
    resized_image = tf.scalar_mul( 1./255,  resized_image )

    return resized_image

def slice_input_producer_demo( data_path , train_file_path ):
    tf.reset_default_graph()

    image_path_list , landmark_list, gender, smile, glasses, pose = load_data_path( data_path , train_file_path )

    train_input_queue = tf.train.slice_input_producer( 
            [image_path_list , landmark_list, gender, smile, glasses, pose] ,
            capacity = 128 )

    image_queue = fetch_image( train_input_queue[0] )
    landmark_queue = train_input_queue[1]
    gender_queue = train_input_queue[2]
    smile_queue = train_input_queue[3]
    glasses_queue = train_input_queue[4]
    pose_queue = train_input_queue[5]

    # generate batch
    image_batch, landmark_batch, gender_batch, smile_batch, glasses_batch, pose_batch = \
    tf.train.shuffle_batch( [image_queue, landmark_queue, gender_queue, smile_queue, 
        glasses_queue, pose_queue ] , batch_size = 32 , capacity = 1000, 
        min_after_dequeue = 10, num_threads = 4, shapes = [(224,224,3) , (10), 
            () , () , () , () ])
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    for i in range(100):
        image, landmark, gender, smile, glasses, pose = sess.run( 
            [image_batch, landmark_batch, gender_batch, smile_batch, glasses_batch, pose_batch] )

        for k in range( 32 ):
            fig = plt.figure()
            fig.add_subplot(1,2,1)
            plt.imshow( image[k] )
            fig.add_subplot(1,2,2)
            plt.imshow( image[k] )
            plt.show()
    coord.request_stop()
    coord.join(threads)
    sess.close()

slice_input_producer_demo( data_path , train_file_path )
