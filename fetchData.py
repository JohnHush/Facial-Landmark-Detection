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

def fetch_numpy_arrays( data_prefix , is_train ):
    """
    the method will load the path in the training.txt or testing.txt files
    
    then load the images and scale them into fixed size, recently, 96(width) * 112(height)
        scale the image so the they will be the same size, it can feed into TF
    """
    if is_train:
        image_path , landmarks, gender, smile, glasses, pose = load_path( data_prefix , True )
    else:
        image_path , landmarks, gender, smile, glasses, pose = load_path( data_prefix , False )

    imgs = list( map( lambda s: cv2.imread(s) , image_path ) )

    def resize_fn( img_landmark_pairs ):
        img = img_landmark_pairs[0]
        lan = img_landmark_pairs[1]

        heigh = 1. * img.shape[0]
        width = 1. * img.shape[1]
        
        lan[ 0:5 ] = lan[ 0:5 ] / width
        lan[ 5:10] = lan[ 5:10] / heigh

        img_resize = cv2.resize( img , ( 96 , 112 ) )
        return img_resize , lan

    # transfer all the landmarks and images into fixed size, landmarks to the range [0, 1]
    imgs_landmarks = list( map( resize_fn , list( zip( imgs , landmarks ) ) ) )
    imgs      = np.array( [ x[0] for x in imgs_landmarks ] )
    landmarks = np.array( [ x[1] for x in imgs_landmarks ] )

    return imgs, landmarks, gender, smile, glasses, pose

def train_input_fn_v2( img_placeholder, l, g, s, gl, p, batch_size = 128 ):
    # the img_placeholder will hold a numpy of imgs
    # the img numpy array is too large so need to be inserted as this way
    dataset = tf.data.Dataset.from_tensor_slices( ( img_placeholder, l, g, s, gl, p ) )
    dataset = dataset.map( input_parser_v2 )
    dataset = dataset.repeat().shuffle( 5 * batch_size ).batch( batch_size )

    return dataset

def evaluate_input_fn_v2( data_prefix, batch_size = 128 ):
    i, l, g, s, gl, p = load_path( data_prefix, False )

    img_np = np.array( list( map( lambda s: cv2.imread( s ) , i ) ) )

    dataset = tf.data.Dataset.from_tensor_slices( ( img_np, l, g, s, gl, p ) )
    dataset = dataset.map( input_parser_v2 )
    dataset = dataset.batch( batch_size )

    return dataset

def input_parser_v2( image, landmarks, gender, smile, glasses, pose ):
    # image augmentation using tf image module
    tf_image = tf.image.random_brightness( image , max_delta = 0.5 )
    tf_image = tf.image.random_contrast( tf_image , 0.2 , 0.7 )

    # don't need to do the resize anymore
    # it's has been done in the outside
    tf_image = tf.to_float( tf_image - 128 )
    tf_image = tf.scalar_mul( 1./255, tf_image )

    return dict( image = tf_image ) , dict ( landmarks = landmarks, 
            gender = gender, smile = smile, glasses = glasses, pose = pose )

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
    image_path , landmarks, gender, smile, glasses, pose = load_path( args.data_path , True )

    def transfer( path ):
        img = cv2.imread( path )
        img_resize = cv2.resize( img , ( 100 , 100 ) )
        return img_resize

    imgs = list( map( transfer , image_path ) )
    imgs = np.array( imgs )
    print( imgs.shape )
    print( imgs[0].shape )
    print( imgs[9999].shape )

    print( landmarks.shape )

    sss = tf.convert_to_tensor( imgs )

    fetch_numpy_arrays( args.data_path , False )

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
