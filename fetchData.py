import cv2
import tensorflow as tf
import numpy as np
import os.path as op
import os
import matplotlib.pyplot as plt
from functools import reduce

class MSCELEB( object ):
    """
    coorperate with tf.TextLineDataset to build the input graph

    meanwhile, several helper function will be added,
        such as: 
        try fetch the image to see the dataset could be loaded or not
        try to illustrate the img to see how it distributes

        or partition the dataset to train and test
        or just use it as train dataset
    """
    def __init__( self, anno_path , data_dir , \
            if_split = True , \
            train_ratio = 0.9 ):
        self._anno_path = anno_path
        self._data_dir  = data_dir
        self._if_split  = if_split
        self._train_ratio = train_ratio
        self._height = 112
        self._width = 96

        _prepareEverything( anno_path , data_dir )
    def _prepareEverything( self , anno_path , data_dir ):
        # read info from annotation file
        anno_info = np.genfromtxt( anno_path, delimiter=" ", unpack=True, \
                dtype= 'str')
        if not _try_fetchIMG( anno_info ):
            print( "the dataset could not be loaded!" )
            return

        self._num_images = len( anno_info[0] )
        if _if_split:
            self._num_train_images = _train_ratio * _num_images
            self._num_test_images  = _num_images - _num_train_images

    def _try_fetchIMG( self , anno_info ):
        """
        in initialization, we make some test in fetching
        """
        img_path = anno_info[0]
        try_num = min( 5 , len( anno_info[0] ) )
        
        for _ in range( try_num ):
            try_index = np.random.randint( 0 , len( anno_info[0] ) -1 )
            try_path = os.path.join( self._data_dir , anno_info[0][try_index] )
            img = cv2.imread( try_path )

            if img == None:
                print( "the img %s doesn't exist" % try_path )
                return False
        return True

    def dataStream( self , batch_size , if_shuffle = True ):
        dataset = tf.data.TextLineDataset( anno_path )
        dataset = dataset.map( _parser )
        if if_shuffle:
            dataset = dataset.shuffle( 10 * batch_size )
        dataset = dataset.repeat().prefetch( 10 * batch_size )
        dataset = dataset.batch( batch_size )

        return dataset

    def trainDataStream( self , batch_size , if_shuffle = True ):
        dataset = tf.data.TextLineDataset( anno_path )
        dataset = dataset.take( _num_train_images )
        dataset = dataset.map( _parser )
        if if_shuffle:
            dataset = dataset.shuffle( 10 * batch_size )
        dataset = dataset.repeat().prefetch( 10 * batch_size )
        dataset = dataset.batch( batch_size )

        return dataset

    def testDataStream( self , batch_size ):
        dataset = tf.data.TextLineDataset( anno_path )
        dataset = dataset.skip( _num_train_images )
        dataset = dataset.map( _parser )
        dataset = dataset.prefetch( 10 * batch_size )
        dataset = dataset.batch( batch_size )

        return dataset
    
    def _parser( line ):
        """
        data line in landmark file in a form:
            data_path ID_number x1 y1 x2 y2 x3 y3 x4 y4 x5 y5
        """
        FIELD_DEFAULT = [ ['IMG_PATH'] , [0], [0.], [0.], [0.], [0.]\
                , [0.], [0.], [0.], [0.], [0.], [0.] ]
        fields = tf.decode_csv( line , FIELD_DEFAULT )
        content = tf.read_file( _data_dir + '/' + fields[0] )

        # transfer the landmark into 
        # x1, x2, x3, x4, x5, y1, y2, y3, y4, y5 format
        landmarks = tf.concat( fields[2], fields[4], fields[6], fields[8]\
                , fields[10], fields[3], fields[5], fields[7], fields[9], \
                fields[11] )
        tf_image = tf.image.decode_jpeg( content , channels = 3 )

        # scale the landmarks in the range 0 to 1
        height = tf.to_double( tf.shape( tf_image )[0] )
        width  = tf.to_double( tf.shape( tf_image )[1] )

        transformed_landmarks = tf.concat( [ landmarks[0:5] / width, \
                landmarks[5:10] / height] , -1 )

        tf_image = tf.image.resize_images( tf_image , [ _height , _width ] )
        tf_image = tf_image - 128
        tf_image = tf.scalar_mul( 1./255,  tf_image )
        tf_image = tf.reshape( tf_image , [ _height, _width , 3 ] )

        # recently not need to return a dict
        # i don't use Estimator here
        return tf_image , transformed_landmarks

        
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
    sess = tf.InteractiveSession()

    ms_data = MSCELEB( anno_path , data_dir )
    iterator = ms_data.dataStream( batch_size = 32 )
    imgs , landmarks = iterator.get_next().run()

    print( landmarks )
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
