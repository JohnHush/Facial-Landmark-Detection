import cv2
import tensorflow as tf
import numpy as np
import os.path as op
import matplotlib.pyplot as plt

class args( object ):
    pass

args.RESIZE_SIZE = 48
args.data_path = "/Users/pitaloveu/working_data/MTFL"

def load_path( data_prefix , if_train = True ):
    # in the train file, 
    # formatting as [ "image path", 10 * landmark(float), 4*character(int) ]
    if if_train:
        file_path = op.join( data_prefix , "training.txt" )
    else:
        file_path = op.join( data_prefix , "testing.txt" )

    # using numpy function to read infor from txt files
    # read data using str format, then reformatting the numeric values
    # into demanding ones using astype
    image_info = np.genfromtxt( file_path, delimiter= " ", unpack=True, dtype = 'str')

    # replace all the '\' in the string by '/'
    image_info[0] = [ image_info[0][i].replace( "\\" , "/" ) 
            for i in range( image_info.shape[1] ) ]

    # filter all unexisting files
    filter_indexes = [ op.isfile( op.join( data_prefix , image_info[0][i] ) ) 
            for i in range(image_info.shape[1]) ]
    #image_info[0] = [ op.join( data_prefix, image_info[0][i] )
    #        for i in range(image_info.shape[1]) ]

    image_info = image_info[:,filter_indexes]

    # fetch data_path, 10 landmarks and 4 characters
    # and reshape , then transpose to feed into 
    # dataset using from_tensor_slices method
    image_path = np.array([ op.join( data_prefix , image_info[0][i] )
        for i in range( image_info.shape[1])] )
    landmarks  = np.transpose( image_info[1:11].astype( float ) )
    gender     = image_info[11].astype( int )
    smile      = image_info[12].astype( int )
    glasses    = image_info[13].astype( int )
    pose       = image_info[14].astype( int )

    return image_path , landmarks, gender, smile, glasses, pose

def fetch_one_batch( data_prefix, if_train = True , batch_size = 32 ):
    i, l, g, s, gl, p = load_path( data_prefix, if_train )

    dataset = tf.data.Dataset.from_tensor_slices( (i, l, g, s, gl, p ) )
    dataset = dataset.map( input_parser )
    dataset = dataset.repeat().batch( batch_size )

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

def input_parser( image_path, landmarks, gender, smile, glasses, pose ):

    content = tf.read_file( image_path )
    tf_image = tf.image.decode_jpeg( content , channels = 3 )
    #resized_image = tf.image.resize_images( tf_image , [ 224 , 224 ] )
    #resized_image = tf.scalar_mul( 1./255,  resized_image )

    #return resized_image, landmarks, gender, smile, glasses, pose 
    return tf_image, landmarks, gender, smile, glasses, pose

if __name__ == "__main__":

    with tf.Session() as sess:
        image, l, g, s, gl, p = sess.run( fetch_one_batch( args.data_path , batch_size = 20 ))
        for k in range(20):
            canvas = image[k]
            landmark = l[k]
            # draw left eye
            x1 = int( landmark[0]) 
            y1 = int(landmark[5])
           
            red = ( 0, 0, 255,)
            blue = ( 255,0,0,)
            green = ( 0,255,0,)
            cv2.circle( canvas , ( x1, y1 ) , 3 , red , -1 )
            cv2.circle( canvas , ( int( landmark[1]), int(landmark[6]) ) , 3 , red , -1 )
            cv2.circle( canvas , ( int( landmark[2]), int(landmark[7]) ) , 3 , blue , -1 )
            cv2.circle( canvas , ( int( landmark[3]), int(landmark[8]) ) , 3 , green , -1 )
            cv2.circle( canvas , ( int( landmark[4]), int(landmark[9]) ) , 3 , green , -1 )
            fig = plt.figure()
            plt.imshow( canvas )
            plt.show()
