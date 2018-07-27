import cv2
import tensorflow as tf
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from functools import reduce

class args( object ):
    pass

args.RESIZE_SIZE = 48
args.data_path = "/Users/pitaloveu/working_data/MTFL"

def write_resized_test_img( data_path , output_dir = "resized" ):
    image_path , landmarks, gender, smile, glasses, pose = \
            load_path( data_path , if_train = False )

    imgs = list( map( lambda s: cv2.imread( s ) , image_path ) )
    imgs_resized = list( map ( lambda s: cv2.resize( s , ( 96 , 112 ) ) , imgs ) )

    imgs_resized_path = list( map( lambda s: s.replace( 'AFLW', output_dir, 1 ) , image_path ) )

    for img , path in list( zip( imgs_resized , imgs_resized_path ) ):
        cv2.imwrite( path , img )

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


def train_input_fn( data_prefix, batch_size = 128 ):

    i, l, g, s, gl, p = load_path( data_prefix, True )

    dataset = tf.data.Dataset.from_tensor_slices( (i, l, g, s, gl, p ) )
    dataset = dataset.map( input_parser )
    dataset = dataset.repeat().shuffle( 5 * batch_size ).batch( batch_size )

    return dataset

def evaluate_input_fn( data_prefix, batch_size = 128 ):

    i, l, g, s, gl, p = load_path( data_prefix, False )

    dataset = tf.data.Dataset.from_tensor_slices( (i, l, g, s, gl, p ) )
    dataset = dataset.map( input_parser )
    dataset = dataset.batch( batch_size )

    return dataset

def input_parser( image_path, landmarks, gender, smile, glasses, pose ):

    content = tf.read_file( image_path )
    tf_image = tf.image.decode_jpeg( content , channels = 3 )

    # image augmentation using tf image module
    # 
    tf_image = tf.image.random_brightness( tf_image , max_delta = 0.5 )
    tf_image = tf.image.random_contrast( tf_image , 0.2 , 0.7 )

    # scale the landmarks in the range 0 to 1
    height = tf.to_double( tf.shape( tf_image )[0] )
    width  = tf.to_double( tf.shape( tf_image )[1] )

    transformed_landmarks = tf.concat( [ landmarks[0:5] / width, \
            landmarks[5:10] / height] , -1 )

    tf_image = tf.image.resize_images( tf_image , [ 112 , 96 ] )
    tf_image = tf_image - 128
    tf_image = tf.scalar_mul( 1./255,  tf_image )
    tf_image = tf.reshape( tf_image , [ 112 , 96 , 3 ] )

    return dict( image = tf_image ) , dict ( landmarks = transformed_landmarks, 
            gender = gender, smile = smile, glasses = glasses, pose = pose )

if __name__ == "__main__":

    write_resized_test_img( args.data_path , output_dir = "resized" )

    """
    iterator = train_eval_input_fn( args.data_path , batch_size = 8 ).make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        image, labels = sess.run( next_element )
        #image, labels = sess.run( train_eval_input_fn( args.data_path , batch_size = 8 ))
        for k in range( 8 ):
            canvas = image['image'][k]
            landmark = labels['landmarks'][k]
            pose = labels['pose'][k]

            print( pose )

            test = canvas[:,:,0]
            # draw left eye
            x1 = int( landmark[0]) 
            y1 = int(landmark[5])
           
            red = ( 0, 0, 255,)
            blue = ( 255,0,0,)
            green = ( 0,255,0,)
            #cv2.circle( canvas , ( x1, y1 ) , 3 , red , -1 )
            #cv2.circle( canvas , ( int( landmark[1]), int(landmark[6]) ) , 3 , red , -1 )
            #cv2.circle( canvas , ( int( landmark[2]), int(landmark[7]) ) , 3 , blue , -1 )
            #cv2.circle( canvas , ( int( landmark[3]), int(landmark[8]) ) , 3 , green , -1 )
            #cv2.circle( canvas , ( int( landmark[4]), int(landmark[9]) ) , 3 , green , -1 )
            fig = plt.figure()
            plt.imshow( test )
            plt.show()
    """
