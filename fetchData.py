import cv2
import tensorflow as tf
import numpy as np
import os.path as op
import os
import matplotlib.pyplot as plt
from functools import reduce
import tempfile

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

        # set the tmp dir to save train anno file and test ano file
        tempfile.tempdir = '/home/public/data/tmp'

        self._prepareEverything( )
    def _prepareEverything( self ):
        # read info from annotation file
        #anno_info = np.genfromtxt( anno_path, delimiter=" ", unpack=True, \
        #        dtype= 'str')
        with open( self._anno_path , 'r' ) as fo:
            try_fetch = fo.readline().strip()
            split_list = try_fetch.split( ' ' )

        if not self._try_fetchIMG( split_list[0] ):
            print( "the dataset could not be loaded!" )
            return

        self._num_images = self._lines_num( self._anno_path )

        if self._if_split:
            self._num_train_images = int( self._train_ratio * self._num_images )
            self._num_test_images  = int( self._num_images - self._num_train_images )
            _ , self._train_file = tempfile.mkstemp( suffix = 'train' )
            _ , self._test_file  = tempfile.mkstemp( suffix = 'test' )

            # split into two files
            with open( self._anno_path , 'r' ) as fr:
                lines = fr.readlines()
                with open( self._train_file , 'w' ) as fw:
                    fw.writelines( lines[ 0: self._num_train_images ] )
                with open( self._test_file , 'w' ) as fw:
                    fw.writelines( lines[ self._num_train_images : ] )

    def _lines_num( self , file ):
        count = 0
        with open( file , 'r' ) as fi:
            while True:
                buffer = fi.read( 1024 * 8092 )
                if not buffer:
                    break
                count += buffer.count( '\n' )
        return count

    def _try_fetchIMG( self , try_fetch_path ):
        """
        in initialization, we make some test in fetching
        """
        try_path = os.path.join( self._data_dir , try_fetch_path )
        img = cv2.imread( try_path )
        if img is None:
            return False
        return True

    def exportTestData( out_dir , out_size = [] ):
        if not self._if_split:
            print( "this dataset hasn't been splitted, it \
                    shouldn't be exported!" )
            return

        with open( self._test_file , 'r' ) as fr:
            lines = fr.readlines()
        for line in lines:
            split_line = line.split( ' ' ).strip()
            img_path = os.path.join( self._data_dir , split_line[0] )
            img = cv2.imread( img_path )
            if len(out_size) == 2:
                # here in opencv , resize function accept :
                # width , height parameters
                # it's a little confusing
                img = cv2.resize( img , ( out_size[1] , out_size[0] ) )

            output_path = os.path.join( out_dir , \
                    img_path[ img_path.rfind('/') + 1:] )

            cv2.imwrite( output_path , img )

    def dataStream( self , batch_size , if_shuffle = True ):
        dataset = tf.data.TextLineDataset( self._anno_path )
        dataset = dataset.map( self._parser )
        if if_shuffle:
            dataset = dataset.shuffle( 10 * batch_size )
        dataset = dataset.repeat().prefetch( 10 * batch_size )
        dataset = dataset.batch( batch_size )

        return dataset.make_one_shot_iterator().get_next()

    def trainDataStream( self , batch_size , if_shuffle = True ):
        dataset = tf.data.TextLineDataset( self._train_file )
        #dataset = dataset.take( self._num_train_images )
        dataset = dataset.map( self._parser )
        if if_shuffle:
            dataset = dataset.shuffle( 10 * batch_size )
        dataset = dataset.repeat().prefetch( 10 * batch_size )
        dataset = dataset.batch( batch_size )

        return dataset.make_one_shot_iterator().get_next()

    def testDataStream( self , batch_size ):
        dataset = tf.data.TextLineDataset( self._test_file )
        #dataset = dataset.skip( self._num_train_images )
        dataset = dataset.map( self._parser )
        dataset = dataset.prefetch( 10 * batch_size )
        dataset = dataset.batch( batch_size )

        return dataset.make_one_shot_iterator().get_next()
    
    def _parser( self , line ):
        """
        data line in landmark file in a form:
            data_path ID_number x1 y1 x2 y2 x3 y3 x4 y4 x5 y5
        """
        FIELD_DEFAULT = [ ['IMG_PATH'] , [0], [0.], [0.], [0.], [0.]\
                , [0.], [0.], [0.], [0.], [0.], [0.] ]
        fields = tf.decode_csv( line , FIELD_DEFAULT , field_delim = ' ' )
        content = tf.read_file( self._data_dir + '/' + fields[0] )

        # transfer the landmark into 
        # x1, x2, x3, x4, x5, y1, y2, y3, y4, y5 format
        landmarks = tf.stack( [ fields[2], fields[4], fields[6], fields[8]\
                , fields[10], fields[3], fields[5], fields[7], fields[9], \
                fields[11] ] , axis = -1  )
        landmarks = tf.to_float( landmarks )
        tf_image = tf.image.decode_jpeg( content , channels = 3 )

        # scale the landmarks in the range 0 to 1
        height = tf.to_float( tf.shape( tf_image )[0] )
        width  = tf.to_float( tf.shape( tf_image )[1] )

        transformed_landmarks = tf.concat( [ landmarks[0:5] / width, \
                landmarks[5:10] / height] , -1 )

        tf_image = tf.image.resize_images( tf_image , [ self._height , self._width ] )
        tf_image = tf_image - 128
        tf_image = tf.scalar_mul( 1./255,  tf_image )
        tf_image = tf.reshape( tf_image , [ self._height, self._width , 3 ] )

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
    
    anno_path__ = '/home/public/data/celebrity_lmk'
    data_dir__ = '/home/public/data'
    ms_data = MSCELEB( anno_path__ , data_dir__ )
    imgs , landmarks =  sess.run( ms_data.testDataStream( batch_size = 10) )

    print( landmarks )
    for i in range( len( imgs ) ):
        cv2.imshow( "" , imgs[i] )
        cv2.waitKey()
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
