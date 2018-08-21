import cv2
import tensorflow as tf
import numpy as np
import os.path as op
import os
import matplotlib.pyplot as plt
from functools import reduce
import tempfile
from CONFIGURES import args

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

        with open( self._anno_path , 'r' ) as fr:
            lines = fr.readlines()

        # make a dict, key is the file name, value is landmark
        self._imgName_landmark_dict = {}
        for line in lines:
            split_line = line.strip().split( ' ' )
            key = os.path.join( self._data_dir , split_line[0] )
            value = [ float(x) for x in split_line[ 2 : ] ]
            value = value[ 0::2 ] + value[ 1::2 ]
            self._imgName_landmark_dict[key] = value

        self._num_images = self._lines_num( self._anno_path )

        if self._if_split:
            self._num_train_images = int( self._train_ratio * self._num_images )
            self._num_test_images  = int( self._num_images - self._num_train_images )
            _ , self._train_file = tempfile.mkstemp( suffix = 'train' )
            _ , self._test_file  = tempfile.mkstemp( suffix = 'test' )
            self._trainImgName_landmark_dict = {}
            self._testImgName_landmark_dict  = {}

            # split into two files
            with open( self._train_file , 'w' ) as fw:
                fw.writelines( lines[ 0: self._num_train_images ] )

                for line in lines[ 0: self._num_train_images ]:
                    split_line = line.strip().split( ' ' )
                    key = os.path.join( self._data_dir , split_line[0] )
                    value = [ float(x) for x in split_line[ 2 : ] ]
                    value = value[ 0::2 ] + value[ 1::2 ]
                    self._trainImgName_landmark_dict[key] = value

            with open( self._test_file , 'w' ) as fw:
                fw.writelines( lines[ self._num_train_images : ] )

                for line in lines[ self._num_train_images : ]:
                    split_line = line.strip().split( ' ' )
                    key = os.path.join( self._data_dir , split_line[0] )
                    value = [ float(x) for x in split_line[ 2 : ] ]
                    value = value[ 0::2 ] + value[ 1::2 ]
                    self._testImgName_landmark_dict[key] = value

    @property
    def imgName_landmark_dict( self ):
        return self._imgName_landmark_dict

    @property
    def trainImgName_landmark_dict( self ):
        if not self._if_split:
            print( "there is no train dict generated cause no splitting!" )
            return {}
        return self._trainImgName_landmark_dict

    @property
    def testImgName_landmark_dict( self ):
        if not self._if_split:
            print( "there is no test dict generated cause no splitting!" )
            return {}
        return self._testImgName_landmark_dict

    def _lines_num( self , file ):
        count = 0
        with open( file , 'r' ) as fi:
            while True:
                buffer = fi.read( 1024 * 8092 )
                if not buffer:
                    break
                count += buffer.count( '\n' )
        return count

    def _computeBBoxOfLandmarks( self , landmarks ):
        x1 = landmarks[0]
        x2 = landmarks[1]
        x3 = landmarks[2]
        x4 = landmarks[3]
        x5 = landmarks[4]

        y1 = landmarks[5]
        y2 = landmarks[6]
        y3 = landmarks[7]
        y4 = landmarks[8]
        y5 = landmarks[9]

        xmax = tf.maximum( x1 , tf.maximum( x2 , tf.maximum( x3 , \
                tf.maximum( x4 , x5 ) )))

        xmin = tf.minimum( x1 , tf.minimum( x2 , tf.minimum( x3 , \
                tf.minimum( x4 , x5 ) )))

        ymax = tf.maximum( y1 , tf.maximum( y2 , tf.maximum( y3 , \
                tf.maximum( y4 , y5 ) )))

        ymin = tf.minimum( y1 , tf.minimum( y2 , tf.minimum( y3 , \
                tf.minimum( y4 , y5 ) )))

        return xmax , xmin , ymax , ymin

    def _clipAndResizeBBox( self , tf_image , landmarks ):
        extend_ratio_X = 0.9
        extend_ratio_Y = 0.9

        height = tf.to_float( tf.shape( tf_image )[0] )
        width  = tf.to_float( tf.shape( tf_image )[1] )

        xmax , xmin , ymax , ymin = self._computeBBoxOfLandmarks( landmarks )

        bbox_width  = xmax - xmin
        bbox_height = ymax - ymin

        clip_x1 = xmin - extend_ratio_X * bbox_width
        clip_x2 = xmax + extend_ratio_X * bbox_width
        clip_y1 = ymin - extend_ratio_Y * bbox_height
        clip_y2 = ymax + extend_ratio_Y * bbox_height

        clip_width = clip_x2 - clip_x1
        clip_height= clip_y2 - clip_y1

        landmarks = tf.concat( [ ( landmarks[0:5] - clip_x1 ) / clip_width , \
                ( landmarks[5:]  - clip_y1 ) / clip_height ] , axis = -1 )

        #param_boxes = tf.constant( [ clip_y1 , clip_x1 , clip_y2 , \
        #        clip_x2 ] , dtype = float , shape = [1,4] )
        param_boxes = tf.expand_dims ( tf.stack( [ clip_y1 , clip_x1 , clip_y2 , \
                clip_x2 ] ) , 0 )
        param_box_ind = tf.constant( [0] , dtype = tf.int32 , shape = [1] )
        param_crop_size = tf.constant( [self._height , self._width] ,\
                dtype = tf.int32 , shape = [2] )

        tf_image = tf.expand_dims( tf_image , 0 )
        tf_image = tf.image.crop_and_resize( tf_image , \
                param_boxes , \
                param_box_ind , \
                param_crop_size )

        return tf_image , landmarks

    def _try_fetchIMG( self , try_fetch_path ):
        """
        in initialization, we make some test in fetching
        """
        try_path = os.path.join( self._data_dir , try_fetch_path )
        img = cv2.imread( try_path )
        if img is None:
            return False
        return True

    def exportTestData( self ,  out_dir , out_size ):
        if not self._if_split:
            print( "this dataset hasn't been splitted, it \
                    shouldn't be exported!" )
            return

        with open( self._test_file , 'r' ) as fr:
            lines = fr.readlines()
        for line in lines:
            split_line = line.strip().split( ' ' )
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
        dataset = dataset.repeat().prefetch( 20 * batch_size )
        dataset = dataset.batch( batch_size )

        return dataset.make_one_shot_iterator().get_next()

    def trainDataStreamClipped( self , batch_size , if_shuffle = True ):
        dataset = tf.data.TextLineDataset( self._train_file )
        #dataset = dataset.take( self._num_train_images )
        dataset = dataset.map( lambda line: self._parser( line , \
                if_clip = True ) )
        if if_shuffle:
            dataset = dataset.shuffle( 10 * batch_size )
        dataset = dataset.repeat().prefetch( 20 * batch_size )
        dataset = dataset.batch( batch_size )

        return dataset.make_one_shot_iterator().get_next()

    def testDataStream( self , batch_size ):
        dataset = tf.data.TextLineDataset( self._test_file )
        #dataset = dataset.skip( self._num_train_images )
        dataset = dataset.map( self._parser )
        dataset = dataset.repeat().prefetch( 10 * batch_size )
        dataset = dataset.batch( batch_size )

        return dataset.make_one_shot_iterator().get_next()
    
    def _parser( self , line , if_clip = False ):
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


        if if_clip:
            tf_image , transformed_landmarks = \
                    self._clipAndResizeBBox( tf_image , transformed_landmarks )

        if not if_clip:
            tf_image = tf.image.resize_images( tf_image , [ self._height , self._width ] )
        tf_image = tf_image - 128
        tf_image = tf.scalar_mul( 1./255,  tf_image )
        tf_image = tf.reshape( tf_image , [ self._height, self._width , 3 ] )

        # recently not need to return a dict
        # i don't use Estimator here
        return tf_image , transformed_landmarks

    def showLandmarks( self , sess , testdataStream ):
        """
        trying to show the images very roughly
            while we need a running session to take all the operations
            and a dataStream with ( images , landmarks ) as output is needed
        
        till now we show the landmarks in a loop tradition
        """
        imgs , landmark = sess.run( testdataStream( batch_size = 500 ) )

        red   = ( 0, 0, 255, )
        blue  = ( 255, 0, 0, )
        green = ( 0, 255, 0, )

        for i in range( len( imgs ) ):
            canvas = imgs[i]
            
            cv2.circle( canvas , ( int( landmark[i][0] * self._width ) , \
                    int( landmark[i][5] * self._height ) ) , 3 , red , -1 )
            cv2.circle( canvas , ( int( landmark[i][1] * self._width ) , \
                    int( landmark[i][6] * self._height ) ) , 3 , red , -1 )
            cv2.circle( canvas , ( int( landmark[i][2] * self._width ) , \
                    int( landmark[i][7] * self._height ) ) , 3 , blue , -1 )
            cv2.circle( canvas , ( int( landmark[i][3] * self._width ) , \
                    int( landmark[i][8] * self._height ) ) , 3 , green , -1 )
            cv2.circle( canvas , ( int( landmark[i][4] * self._width ) , \
                    int( landmark[i][9] * self._height ) ) , 3 , green , -1 )
            
            cv2.imshow( "" , canvas )
            cv2.waitKey( 0 )
            #fig = plt.figure()
            #plt.imshow( canvas )
class MTFL( object ):
    """
    contains 12,995 images with annotated with:
        1. five facial landmarks
        2. attributes: gender, smile , glasses, pose 

    wanna details of this dataset, please refer to 
    ECCV 2014 paper "Facial Landmark Detection by Deep Multi-task Learning"
    """
    def __init__( self , data_path  , config ):
        self._data_path = data_path
        self._height = config.img_height
        self._width  = config.img_width
        self.args = config

        tempfile.tempdir = os.path.join( self._data_path , 'tmp' )

        self._prepareEverything()
    
    def _prepareEverything( self ):
        _ , self._train_file = tempfile.mkstemp( suffix = 'train' )
        _ , self._test_file  = tempfile.mkstemp( suffix = 'test' )

        ori_train_file = os.path.join( self._data_path , "training.txt" )
        ori_test_file  = os.path.join( self._data_path , "testing.txt" )

        with open( ori_train_file , 'r' ) as fr:
            lines = fr.readlines()
            lines = [ line.strip( ' ' ) for line in lines if line.strip( ' ' ) != '' ]
            lines = list( map( lambda s: s.replace('\\' , '/') , lines ) )

            with open( self._train_file , 'w' ) as fw:
                fw.writelines( lines )

        with open( ori_test_file , 'r' ) as fr:
            lines = fr.readlines()
            lines = [ line.strip( ' ' ) for line in lines if line.strip(' ') != '' ]
            lines = list( map( lambda s: s.replace('\\' , '/') , lines ) )

            with open( self._test_file , 'w' ) as fw:
                fw.writelines( lines )

    def _poseFilter( self , line , pose ):
        """
        used by testDataStreamFilteredByPose
        """
        FIELD_DEFAULT = [ ['IMAGE_PATH'] , [0.], [0.],[0.],[0.],[0.],[0.],\
                [0.],[0.],[0.],[0.], [0], [0] , [0] , [0] ]

        fields = tf.decode_csv( line , FIELD_DEFAULT , field_delim = ' ' )

        return tf.equal( fields[-1] , pose )

    def _parser(    self , \
                    line , \
                    if_train ):
        """
        data line in annotation file in a form:
          relative_data_path  x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 #gender #smile #glass #pose

        augmentation will include
        """

        FIELD_DEFAULT = [ ['IMAGE_PATH'] , [0.], [0.],[0.],[0.],[0.],[0.],\
                [0.],[0.],[0.],[0.], [0], [0] , [0] , [0] ]

        fields = tf.decode_csv( line , FIELD_DEFAULT , field_delim = ' ' )
        content = tf.read_file( self._data_path + '/' + fields[0] )

        landmarks = tf.stack( [ fields[1], fields[2], fields[3], fields[4], \
                fields[5], fields[6], fields[7], fields[8], fields[9],\
                fields[10] ] , axis = -1 )

        tf_image = tf.image.decode_jpeg( content , channels = 3 )
        if if_train:
            tf_image = tf.image.random_brightness( tf_image , max_delta = 0.5 )
            tf_image = tf.image.random_contrast( tf_image , 0.2 , 0.7 )

        height = tf.to_float( tf.shape( tf_image )[0] )
        width  = tf.to_float( tf.shape( tf_image )[1] )

        transformed_landmarks = tf.concat( [ landmarks[0:5] / width, \
                landmarks[5:10] / height] , -1 )

        """
        if if_aligned:
            tf_image , transformed_landmarks = \
                    self._alignLandmark( tf_image , transformed_landmarks )
        if if_clip:
            tf_image , transformed_landmarks = \
                    self._clipAndResizeBBox( tf_image , transformed_landmarks )

        #tf_image = tf.image.resize_images( tf_image , [ self._height , self._width ] \
        #        , method = tf.image.ResizeMethod.NEAREST_NEIGHBOR )
        if not if_clip or not if_aug :
            tf_image = tf.image.resize_images( tf_image , [ self._height , self._width ])
        #tf_image = tf.to_float( tf_image )

        if if_aug:
            tf_image , transformed_landmarks = \
                    self._augmentation( tf_image , transformed_landmarks )

        #tf_image = tf_image - 128
        tf_image = tf.scalar_mul( 1./255,  tf_image )
        tf_image = tf.reshape( tf_image , [ self._height, self._width , 3 ] )
        """
        return tf_image , transformed_landmarks

    def _normalize_reshape( self , tf_image , landmarks ):
        #tf_image = tf_image - 128
        tf_image = tf.scalar_mul( 1./255,  tf_image )
        tf_image = tf.reshape( tf_image , [ self._height, self._width , 3 ] )
        return tf_image , landmarks

    def _resize( self , tf_image , landmarks ):
        tf_image = tf.image.resize_images( tf_image , [ self._height , self._width ])
        return tf_image , landmarks

    def _computeInnerBoundary( self , xmin, xmax, ymin, ymax ):
        bbox_width  = xmax - xmin
        bbox_heigth = ymax - ymin

        new_xmin = xmin - self.args.inner_left_adding * bbox_width
        new_xmax = xmax + self.args.inner_right_adding * bbox_width
        new_ymin = ymin - self.args.inner_up_adding * bbox_heigth
        new_ymax = ymax + self.args.inner_down_adding * bbox_heigth

        return new_xmin, new_xmax, new_ymin, new_ymax
    
    def _computeOuterBoundary( self , xmin , xmax , ymin , ymax ):
        bbox_width  = xmax - xmin
        bbox_heigth = ymax - ymin

        new_xmin = xmin - self.args.outer_left_adding * bbox_width
        new_xmax = xmax + self.args.outer_right_adding * bbox_width
        new_ymin = ymin - self.args.outer_up_adding * bbox_heigth
        new_ymax = ymax + self.args.outer_down_adding * bbox_heigth

        return new_xmin, new_xmax, new_ymin, new_ymax
    
    def _computeBBoxOfLandmarks( self , landmarks ):
        x1 = landmarks[0]
        x2 = landmarks[1]
        x3 = landmarks[2]
        x4 = landmarks[3]
        x5 = landmarks[4]

        y1 = landmarks[5]
        y2 = landmarks[6]
        y3 = landmarks[7]
        y4 = landmarks[8]
        y5 = landmarks[9]

        xmax = tf.maximum( x1 , tf.maximum( x2 , tf.maximum( x3 , \
                tf.maximum( x4 , x5 ) )))

        xmin = tf.minimum( x1 , tf.minimum( x2 , tf.minimum( x3 , \
                tf.minimum( x4 , x5 ) )))

        ymax = tf.maximum( y1 , tf.maximum( y2 , tf.maximum( y3 , \
                tf.maximum( y4 , y5 ) )))

        ymin = tf.minimum( y1 , tf.minimum( y2 , tf.minimum( y3 , \
                tf.minimum( y4 , y5 ) )))

        return xmax , xmin , ymax , ymin

    def _augmentation( self , tf_image , landmarks ):
        xmax , xmin , ymax , ymin = self._computeBBoxOfLandmarks( landmarks )
        inner_xmin , inner_xmax , inner_ymin , inner_ymax = \
                self._computeInnerBoundary( xmin, xmax , ymin , ymax )

        outer_xmin, outer_xmax, outer_ymin, outer_ymax = \
                self._computeOuterBoundary( xmin, xmax , ymin , ymax )

        bbox_xmin = tf.random_uniform( [1] , outer_xmin, inner_xmin, \
                dtype= tf.float32)[0]

        bbox_xmax = tf.random_uniform( [1] , inner_xmax, outer_xmax, \
                dtype= tf.float32)[0]

        bbox_ymin = tf.random_uniform( [1] , outer_ymin, inner_ymin, \
                dtype= tf.float32)[0]

        bbox_ymax = tf.random_uniform( [1] , inner_ymax, outer_ymax, \
                dtype= tf.float32)[0]

        rotate_angle = tf.random_uniform( [1] , self.args.angle_down , \
                self.args.angle_up , dtype = tf.float32 )[0]

        tf_image , landmarks = self._rotateImg( tf_image , landmarks , rotate_angle )

        height = tf.to_float( tf.shape( tf_image )[0] )
        width  = tf.to_float( tf.shape( tf_image )[1] )

        left_up_corner_x , left_up_corner_y = self._project_pt(  bbox_xmin , \
                bbox_ymin , width, height , rotate_angle )
        right_down_corner_x , right_down_corner_y = self._project_pt(  bbox_xmax , \
                bbox_ymax , width, height , rotate_angle )
        right_up_corner_x , right_up_corner_y = self._project_pt(  bbox_xmax , \
                bbox_ymin , width, height , rotate_angle )
        left_down_corner_x , left_down_corner_y = self._project_pt(  bbox_xmin , \
                bbox_ymax , width, height , rotate_angle )

        clip_x1 = tf.minimum( left_up_corner_x, tf.minimum( right_down_corner_x , \
                tf.minimum( right_up_corner_x , left_down_corner_x ) ))
        clip_x2 = tf.maximum( left_up_corner_x, tf.maximum( right_down_corner_x , \
                tf.maximum( right_up_corner_x , left_down_corner_x ) ))
        clip_y1 = tf.minimum( left_up_corner_y, tf.minimum( right_down_corner_y, \
                tf.minimum( right_up_corner_y , left_down_corner_y ) ))
        clip_y2 = tf.maximum( left_up_corner_y, tf.maximum( right_down_corner_y, \
                tf.maximum( right_up_corner_y , left_down_corner_y ) ))

        clip_width = clip_x2 - clip_x1
        clip_height= clip_y2 - clip_y1

        landmarks = tf.concat( [ ( landmarks[0:5] - clip_x1 ) / clip_width , \
                ( landmarks[5:]  - clip_y1 ) / clip_height ] , axis = -1 )

        param_boxes = tf.expand_dims ( tf.stack( [ clip_y1 , clip_x1 , clip_y2 , \
                clip_x2 ] ) , 0 )
        param_box_ind = tf.constant( [0] , dtype = tf.int32 , shape = [1] )
        param_crop_size = tf.constant( [self._height , self._width] ,\
                dtype = tf.int32 , shape = [2] )

        tf_image = tf.expand_dims( tf_image , 0 )
        tf_image = tf.image.crop_and_resize( tf_image , \
                param_boxes , \
                param_box_ind , \
                param_crop_size )

        return tf_image , landmarks

    def _clipAndResizeBBox( self , tf_image , landmarks ):
        extend_ratio_X = 0.9
        extend_ratio_Y = 0.9

        height = tf.to_float( tf.shape( tf_image )[0] )
        width  = tf.to_float( tf.shape( tf_image )[1] )

        xmax , xmin , ymax , ymin = self._computeBBoxOfLandmarks( landmarks )

        bbox_width  = xmax - xmin
        bbox_height = ymax - ymin

        clip_x1 = xmin - extend_ratio_X * bbox_width
        clip_x2 = xmax + extend_ratio_X * bbox_width
        clip_y1 = ymin - extend_ratio_Y * bbox_height
        clip_y2 = ymax + extend_ratio_Y * bbox_height

        clip_width = clip_x2 - clip_x1
        clip_height= clip_y2 - clip_y1

        landmarks = tf.concat( [ ( landmarks[0:5] - clip_x1 ) / clip_width , \
                ( landmarks[5:]  - clip_y1 ) / clip_height ] , axis = -1 )

        #param_boxes = tf.constant( [ clip_y1 , clip_x1 , clip_y2 , \
        #        clip_x2 ] , dtype = float , shape = [1,4] )
        param_boxes = tf.expand_dims ( tf.stack( [ clip_y1 , clip_x1 , clip_y2 , \
                clip_x2 ] ) , 0 )
        param_box_ind = tf.constant( [0] , dtype = tf.int32 , shape = [1] )
        param_crop_size = tf.constant( [self._height , self._width] ,\
                dtype = tf.int32 , shape = [2] )

        tf_image = tf.expand_dims( tf_image , 0 )
        tf_image = tf.image.crop_and_resize( tf_image , \
                param_boxes , \
                param_box_ind , \
                param_crop_size )

        return tf_image , landmarks


    def _computeAngle( self , landmarks ):
        """
        use the coordinates of two eyes to compute the rotation angle
        actually, the rotation is determined by the TensorFlow Api
        the central point is set to be the rotation center in default
        """
        dx = landmarks[1] - landmarks[0]
        dy = landmarks[6] - landmarks[5]

        # returned theta in [ -pi , pi ]
        # fullfil  x = r * cos( theta )
        #           y = r * sin( theta )
        return tf.atan2( dy , dx )

    def _project_pt( self , x , y , width, height , theta ):
        """
        x , y is normalized by width and height , respectively

        width and height in pixels
        theta in degree

        """
        theta = theta * 3.1415926 / 180
        projective_matrix = tf.contrib.image.angles_to_projective_transforms( \
                -theta , height , width )

        a0 = projective_matrix[0][0]
        a1 = projective_matrix[0][1]
        a2 = projective_matrix[0][2]
        b0 = projective_matrix[0][3]
        b1 = projective_matrix[0][4]
        b2 = projective_matrix[0][5]
        c0 = projective_matrix[0][6]
        c1 = projective_matrix[0][7]

        x = x * width
        y = y * height

        new_x = ( a0 * x + a1 * y + a2 )/( c0 * x + c1 * y + 1 )
        new_y = ( b0 * x + b1 * y + b2 )/( c0 * x + c1 * y + 1 )

        new_x = new_x / width
        new_y = new_y / height

        return new_x , new_y

    def _rotateImg( self , tf_image , landmarks , angle ):
        """
        the line contains: image_TENSORS , landmarks_TENSORS
        first we only rotate the image by testing usage

        angle in degree
        converts it to radian by: angle * 3.1415926 / 180 
        """

        height = tf.to_float( tf.shape( tf_image )[0] )
        width  = tf.to_float( tf.shape( tf_image )[1] )

        #theta = self._computeAngle( landmarks )
        
        theta = angle * 3.1415926 / 180
        inverse_projective_matrix = tf.contrib.image.angles_to_projective_transforms( \
                theta , height , width )
        projective_matrix = tf.contrib.image.angles_to_projective_transforms( \
                -theta , height , width )

        tf_image = tf.contrib.image.transform( tf_image , inverse_projective_matrix )
   
        a0 = projective_matrix[0][0]
        a1 = projective_matrix[0][1]
        a2 = projective_matrix[0][2]
        b0 = projective_matrix[0][3]
        b1 = projective_matrix[0][4]
        b2 = projective_matrix[0][5]
        c0 = projective_matrix[0][6]
        c1 = projective_matrix[0][7]

        landmarks = tf.concat( [ landmarks[0:5] * width, \
                landmarks[5:10] * height] , -1 )

        landmarks =tf.concat( [ ( a0 * landmarks[0:5] + a1 * landmarks[5:] + a2 ) / \
                ( c0 * landmarks[0:5] + c1 * landmarks[5:] + 1 )  , \
                ( b0 * landmarks[0:5] + b1 * landmarks[5:] + b2 ) / \
                ( c0 * landmarks[0:5] + c1 * landmarks[5:] + 1 ) ] , axis = -1 )

        landmarks = tf.concat( [ landmarks[0:5] / width, \
                landmarks[5:10] / height] , -1 )

        return tf_image , landmarks

    def _alignLandmark( self , tf_image , landmarks ):
        """
        the line contains: image_TENSORS , landmarks_TENSORS
        first we only rotate the image by testing usage
        """

        theta = self._computeAngle( landmarks )
        tf_image , landmarks = self._rotateImg( \
                tf_image , landmarks , theta * 180./ 3.1415926 )

    def trainDataStream( self ):
        """
        data stream output original image augmented with different
        brightness and contrast

        lastly resize the image to specified size
        """
        dataset = tf.data.TextLineDataset( self._train_file )
        dataset = dataset.map( \
                lambda line : self._parser( line , if_train = True ) )
        dataset = dataset.map( self._resize )
        dataset = dataset.map( self._normalize_reshape )
        if self.args.if_shuffle:
            dataset = dataset.shuffle( 3 * self.args.train_batch_size )
        dataset = dataset.repeat().prefetch( 20 * self.args.train_batch_size )
        dataset = dataset.batch( self.args.train_batch_size )

        return dataset.make_one_shot_iterator().get_next()

    def trainDataStreamAugmented( self ):
        dataset = tf.data.TextLineDataset( self._train_file )
        dataset = dataset.map( \
                lambda line : self._parser( line , if_train = False ))
        dataset = dataset.map( self._augmentation )
        dataset = dataset.map( self._normalize_reshape )
        batch_size = self.args.train_batch_size
        if self.args.if_shuffle:
            dataset = dataset.shuffle( 3 * batch_size )
        dataset = dataset.repeat().prefetch( 20 * batch_size )
        dataset = dataset.batch( batch_size )

        return dataset.make_one_shot_iterator().get_next()

    def testDataStream( self, batch_size ):
        dataset = tf.data.TextLineDataset( self._test_file )
        dataset = dataset.map( lambda line : self._parser( line , False) )
        dataset = dataset.repeat().prefetch( 10 * batch_size )
        dataset = dataset.batch( batch_size )

        return dataset.make_one_shot_iterator().get_next()

    def testDataStreamAligned( self, batch_size ):
        dataset = tf.data.TextLineDataset( self._test_file )
        dataset = dataset.map( lambda line : self._parser( line , False,\
                if_aligned = True) )
        dataset = dataset.repeat().prefetch( 10 * batch_size )
        dataset = dataset.batch( batch_size )

        return dataset.make_one_shot_iterator().get_next()

    def testDataStreamAlignedAndClipped( self, batch_size ):
        dataset = tf.data.TextLineDataset( self._test_file )
        dataset = dataset.map( lambda line : self._parser( line , False,\
                if_aligned = True , if_clip = True ) )
        dataset = dataset.repeat().prefetch( 10 * batch_size )
        dataset = dataset.batch( batch_size )

        return dataset.make_one_shot_iterator().get_next()

    def testDataStreamFilteredByPose( self, batch_size , pose ):
        dataset = tf.data.TextLineDataset( self._test_file )
        dataset = dataset.filter( lambda line: self._poseFilter( line , pose ) )
        dataset = dataset.map( lambda line : self._parser( line , False) )
        dataset = dataset.repeat().prefetch( 10 * batch_size )
        dataset = dataset.batch( batch_size )

        return dataset.make_one_shot_iterator().get_next()

    def showLandmarks( self , sess , testdataStream ):
        """
        trying to show the images very roughly
            while we need a running session to take all the operations
            and a dataStream with ( images , landmarks ) as output is needed
        
        till now we show the landmarks in a loop tradition
        """
        imgs , landmark = sess.run( testdataStream() )

        red   = ( 0, 0, 255, )
        blue  = ( 255, 0, 0, )
        green = ( 0, 255, 0, )

        for i in range( len( imgs ) ):
            canvas = imgs[i]
            
            cv2.circle( canvas , ( int( landmark[i][0] * self._width ) , \
                    int( landmark[i][5] * self._height ) ) , 3 , red , -1 )
            cv2.circle( canvas , ( int( landmark[i][1] * self._width ) , \
                    int( landmark[i][6] * self._height ) ) , 3 , red , -1 )
            cv2.circle( canvas , ( int( landmark[i][2] * self._width ) , \
                    int( landmark[i][7] * self._height ) ) , 3 , blue , -1 )
            cv2.circle( canvas , ( int( landmark[i][3] * self._width ) , \
                    int( landmark[i][8] * self._height ) ) , 3 , green , -1 )
            cv2.circle( canvas , ( int( landmark[i][4] * self._width ) , \
                    int( landmark[i][9] * self._height ) ) , 3 , green , -1 )
            
            cv2.imshow( "" , canvas )
            cv2.waitKey( 0 )
            #fig = plt.figure()
            #plt.imshow( canvas )

    def exportTestData( self , out_dir , out_size ):
        """
        this function is built originally for MTCNN test
        we export the specified sized images to reduce the computation
        in reading the original images repeatedly

        ANYWAY, this could be deprecated
        """
        test_file = os.path.join( self._data_path , "testing.txt" )

        with open( test_file , 'r' ) as fr:
            lines = fr.readlines()
        for line in lines:
            split_line = line.strip().split( ' ' )
            split_line[0] = split_line[0].replace( '\\' , '/' )
            img_path = os.path.join( self._data_path , split_line[0] )
            img = cv2.imread( img_path )
            if len(out_size) == 2:
                # here in opencv , resize function accept :
                # width , height parameters
                # it's a little confusing
                img = cv2.resize( img , ( out_size[1] , out_size[0] ) )

            output_path = os.path.join( out_dir , \
                    img_path[ img_path.rfind('/') + 1:] )

            cv2.imwrite( output_path , img )

if __name__ == "__main__":
    sess = tf.InteractiveSession()
   
    data_path = "/Users/pitaloveu/working_data/MTFL"
    #data_path = '/home/jh/working_data/MTFL'
    #ms_data = MSCELEB( '/home/public/data/celebrity_lmk' , \
    #        '/home/public/data' )
    ms_data = MTFL( data_path , args )
    #ms_data.exportTestData( '/home/public/data/tmp/testdata' ,\
    #        [112, 96 ] )

    ms_data.showLandmarks(  sess , ms_data.trainDataStreamAugmented )
    #ms_data.showLandmarks(  sess , ms_data.trainDataStream )
