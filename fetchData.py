import cv2
import tensorflow as tf
import numpy as np
import os.path as op
import os
import matplotlib.pyplot as plt
from functools import reduce
import tempfile
from CONFIGURES import args

class Data( object ):
    """
    Base class of Data
    """
    def __init__( self , config ):
        self._config = config
        self._prepareEverything()

    def _prepareEverything( self ):
        """
        this implicit function should implement
        everything needed in preparing the dataset

        such as:
            for MSCELEB dataset, you may need to split the dataset
            by yourself, and some tempfiles may be needed
        """
        raise NotImplementedError

    def _preprocessTrainDataset( self ):
        raise NotImplementedError

    def trainDataStream( self ):
        """
        should build a tensorflow graph to export an operation
        which gives a batch of data when run once in a session
        """
        dataset = self._preprocessTrainDataset()

        return dataset.make_one_shot_iterator().get_next()

    def _preprocessTestDataset( self ):
        raise NotImplementedError

    def testDataStream( self ):
        """
        should build a tensorflow graph to export an operation
        which gives a batch of data when run once in a session
        """
        dataset = self._preprocessTestDataset()

        return dataset.make_one_shot_iterator().get_next()

    @property
    def width(self):
        raise NotImplementedError

    @property
    def height(self):
        raise NotImplementedError

    def showBatch( self , sess , stream ):
        """
        show one batch images and landmarks
        to check everything works well

        when the stream is runned in a Session, it outputs
            images , landmarks
        """
        imgs , landmark = sess.run( stream() )

        lm = np.zeros_like( landmark )
        lm[:, 0:5 ] = landmark[: , 0:5 ] * self.width
        lm[:, 5:  ] = landmark[: , 5:  ] * self.height

        red   = ( 0, 0, 255, )
        blue  = ( 255, 0, 0, )
        green = ( 0, 255, 0, )

        for i in range( len( imgs ) ):
            canvas = imgs[i]

            cv2.circle( canvas , (lm[i][0] , lm[i][5]) , 3 , red , -1 )
            cv2.circle( canvas , (lm[i][1] , lm[i][6]) , 3 , red , -1 )
            cv2.circle( canvas , (lm[i][2] , lm[i][7]) , 3 , blue , -1 )
            cv2.circle( canvas , (lm[i][3] , lm[i][8]) , 3 , green , -1 )
            cv2.circle( canvas , (lm[i][4] , lm[i][9]) , 3 , green , -1 )

            plt.imshow( canvas )
            plt.show()


    # some map function to augment the dataset or do some transformation
    # including:
    #   1. brightness tuning
    #   2. contrast tuning
    #   3. normalization
    #   4. resize

    def _brightness_map( self , img, landmark ):
        return tf.image.random_brightness( img , max_delta=0.01 ) , landmark

    def _contrast_map( self , img, landmark ):
        return tf.image.random_contrast( img , 0.2, 0.7 ) , landmark

    def _normalize_map( self , img , landmark ):
        return tf.image.convert_image_dtype( img , tf.float32), landmark

    def _resize_map( self , img , landmark ):
        return tf.image.resize_images( img , [ self.height , self.width ]) , landmark

    # some landmarks related helper function for the inheritance Classes
    # including:
    #   1. computing the bounding box of a given landmark coordinates
    #   2. project a pt w.r.t. some rotation parameters
    #   3. rotate an image

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

class MSCELEB( Data ):
    """
    coorperate with tf.TextLineDataset to build the input graph

    meanwhile, several helper function will be added,
        such as: 
        or partition the dataset to train and test
        or just use it as train dataset
    """
    def __init__( self , config ):
        super( MSCELEB , self ).__init__( config )

    def _prepareEverything( self ):
        """
        """
        self._anno_path  = self._config.anno_path
        self._train_ratio = self._config.train_ratio
        self._data_dir = self._config.data_path
        self._height = self._config.img_height
        self._width  = self._config.img_width

        tempfile.tempdir = self._config.tempdir_path
        if not os.path.exists( tempfile.tempdir ):
            os.makedirs( tempfile.tempdir )

        self._train_file = os.path.join( tempfile.tempdir , "training.txt" )
        self._test_file  = os.path.join( tempfile.tempdir , "testing.txt" )

        if os.path.exists( self._train_file ) and \
                os.path.exists( self._test_file ):
            self._num_images = self._lines_num( self._anno_path )
            self._num_train_images = int( self._train_ratio * self._num_images )
            self._num_test_images = self._num_images - self._num_train_images
            return

        with open( self._anno_path , 'r' ) as fr:
            lines = fr.readlines()

        self._num_images = len( lines )
        self._num_train_images = int( self._train_ratio * self._num_images )
        self._num_test_images = self._num_images - self._num_train_images

        with open( self._train_file , 'w' ) as fw:
            fw.writelines( lines[ 0: self._num_train_images ] )

        with open( self._test_file , 'w' ) as fw:
            fw.writelines( lines[self._num_train_images : ] )


    def _lines_num( self , file ):
        count = 0
        with open( file , 'r' ) as fi:
            while True:
                buffer = fi.read( 1024 * 8092 )
                if not buffer:
                    break
                count += buffer.count( '\n' )
        return count

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

        return tf_image , transformed_landmarks

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def _preprocessTestDataset( self ):
        """
        testing data preprocessing

        the operations may be done to the original dataset including:
            1. resize ( required )
            2. normalize ( optional, if specified, the value will be transferred
            into the range(0,1) )
        """
        dataset = tf.data.TextLineDataset( self._test_file )
        dataset = dataset.map( self._parser )

        if self._config.if_normalize:
            dataset = dataset.map( self._normalize_map )

        # resize is a required operation
        dataset = dataset.map( self._resize_map )

        dataset = dataset.repeat().prefetch( 20 * self._config.test_batch_size )
        dataset = dataset.batch( self._config.test_batch_size )

        return dataset

    def _preprocessTrainDataset( self ):
        """
        training data preprocessing

        the operations may be done to the original dataset including:
            1. resize ( required )
            2. normalize ( optional, if specified, the value will be transferred
            into the range(0,1) )
            3. brightness setting( optional in training )
            4. contrast setting( optional in training )
            5. clip ( optional in training )
            6. rotate ( optional in training )
            7. shuffle ( optional in training )
        """
        dataset = tf.data.TextLineDataset( self._train_file )
        dataset = dataset.map( self._parser )

        # if specified shuffle
        if self._config.if_shuffle:
            dataset = dataset.shuffle( 3 * self._config.train_batch_size )

        # if specifed brightness tuning
        if self._config.if_bright:
            dataset = dataset.map( self._brightness_map )

        if self._config.if_contrast:
            dataset = dataset.map( self._contrast_map )

        if self._config.if_augmentation:
            dataset = dataset.map( self._augmentation_map )

        if self._config.if_normalize:
            dataset = dataset.map( self._normalize_map )

        # resize is a required operation
        dataset = dataset.map( self._resize_map )

        dataset = dataset.repeat().prefetch( 20 * self._config.train_batch_size )
        dataset = dataset.batch( self._config.train_batch_size )

        return dataset

    def _augmentation_map( self , tf_image , landmarks ):
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

        rotate_angle = tf.random_uniform( [1] , self._config.angle_down , \
                                          self._config.angle_up , dtype = tf.float32 )[0]

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

        clip_x1 = tf.maximum( 0. , clip_x1 )
        clip_y1 = tf.maximum( 0. , clip_y1 )
        clip_x2 = tf.minimum( clip_x2 , 1. )
        clip_y2 = tf.minimum( clip_y2 , 1. )

        clip_width = clip_x2 - clip_x1
        clip_height= clip_y2 - clip_y1

        offset_height = tf.to_int32 ( clip_y1 * height )
        offset_width  = tf.to_int32 ( clip_x1 * height )
        target_height = tf.to_int32 ( clip_height * height )
        target_width  = tf.to_int32 ( clip_width * width )

        landmarks = tf.concat( [ ( landmarks[0:5] - clip_x1 ) / clip_width , \
                                 ( landmarks[5:]  - clip_y1 ) / clip_height ] , axis = -1 )

        tf_image = tf.image.crop_to_bounding_box( tf_image , offset_height, \
                                                  offset_width , target_height , target_width )

        return tf_image , landmarks

    def _computeInnerBoundary( self , xmin, xmax, ymin, ymax ):
        bbox_width  = xmax - xmin
        bbox_heigth = ymax - ymin

        new_xmin = xmin - self._config.inner_left_adding * bbox_width
        new_xmax = xmax + self._config.inner_right_adding * bbox_width
        new_ymin = ymin - self._config.inner_up_adding * bbox_heigth
        new_ymax = ymax + self._config.inner_down_adding * bbox_heigth

        return new_xmin, new_xmax, new_ymin, new_ymax

    def _computeOuterBoundary( self , xmin , xmax , ymin , ymax ):
        bbox_width  = xmax - xmin
        bbox_heigth = ymax - ymin

        new_xmin = xmin - self._config.outer_left_adding * bbox_width
        new_xmax = xmax + self._config.outer_right_adding * bbox_width
        new_ymin = ymin - self._config.outer_up_adding * bbox_heigth
        new_ymax = ymax + self._config.outer_down_adding * bbox_heigth

        return new_xmin, new_xmax, new_ymin, new_ymax

class MTFL( Data ):
    """
    contains 12,995 images with annotated with:
        1. five facial landmarks
        2. attributes: gender, smile , glasses, pose 

    seeking details of this dataset, please refer to
    ECCV 2014 paper "Facial Landmark Detection by Deep Multi-task Learning"
    """
    def __init__( self , config ):
        super( MTFL , self ).__init__( config )

    def _prepareEverything( self ):
        """
        in this preparing function, we mainly transfer the original
        training.txt and testing.txt files in MTFL dataset into UNIX-friendly
        format
            1. delete the last blank line in the file
            2. transfer the mark '\' into '/'
            3. generate the new files in a tempary directory
        """
        self._data_path = self._config.data_path
        self._height = self._config.img_height
        self._width  = self._config.img_width

        tempfile.tempdir = self._config.tempdir_path
        if not os.path.exists( tempfile.tempdir ):
            os.makedirs( tempfile.tempdir )

        self._train_file = os.path.join( tempfile.tempdir , "training.txt" )
        self._test_file  = os.path.join( tempfile.tempdir , "testing.txt" )

        ori_train_file = os.path.join( self._data_path , "training.txt" )
        ori_test_file  = os.path.join( self._data_path , "testing.txt" )

        with open( ori_train_file , 'r' ) as fr:
            lines = fr.readlines()
            lines = [ line.strip( ' ' ) for line in lines if line.strip( ' ' ) != '' ]
            lines = list( map( lambda s: s.replace('\\' , '/') , lines ) )

            if not os.path.exists( self._train_file ):
                with open( self._train_file , 'w' ) as fw:
                    fw.writelines( lines )

        with open( ori_test_file , 'r' ) as fr:
            lines = fr.readlines()
            lines = [ line.strip( ' ' ) for line in lines if line.strip(' ') != '' ]
            lines = list( map( lambda s: s.replace('\\' , '/') , lines ) )

            if not os.path.exists( self._test_file ):
                with open( self._test_file , 'w' ) as fw:
                    fw.writelines( lines )

    def _preprocessTestDataset( self ):
        """
        testing data preprocessing

        the operations may be done to the original dataset including:
            1. resize ( required )
            2. normalize ( optional, if specified, the value will be transferred
            into the range(0,1) )
        """
        dataset = tf.data.TextLineDataset( self._test_file )
        dataset = dataset.map( self._parser )

        if self._config.if_normalize:
            dataset = dataset.map( self._normalize_map )

        # resize is a required operation
        dataset = dataset.map( self._resize_map )

        dataset = dataset.repeat().prefetch( 20 * self._config.test_batch_size )
        dataset = dataset.batch( self._config.test_batch_size )

        return dataset

    def _preprocessTrainDataset( self ):
        """
        training data preprocessing

        the operations may be done to the original dataset including:
            1. resize ( required )
            2. normalize ( optional, if specified, the value will be transferred
            into the range(0,1) )
            3. brightness setting( optional in training )
            4. contrast setting( optional in training )
            5. clip ( optional in training )
            6. rotate ( optional in training )
            7. shuffle ( optional in training )
        """
        dataset = tf.data.TextLineDataset( self._train_file )
        dataset = dataset.map( self._parser )

        # if specified shuffle
        if self._config.if_shuffle:
            dataset = dataset.shuffle( 3 * self._config.train_batch_size )

        # if specifed brightness tuning
        if self._config.if_bright:
            dataset = dataset.map( self._brightness_map )

        if self._config.if_contrast:
            dataset = dataset.map( self._contrast_map )

        if self._config.if_augmentation:
            dataset = dataset.map( self._augmentation_map )

        if self._config.if_normalize:
            dataset = dataset.map( self._normalize_map )

        # resize is a required operation
        dataset = dataset.map( self._resize_map )

        dataset = dataset.repeat().prefetch( 20 * self._config.train_batch_size )
        dataset = dataset.batch( self._config.train_batch_size )

        return dataset

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def _augmentation_map( self , tf_image , landmarks ):
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

        rotate_angle = tf.random_uniform( [1] , self._config.angle_down , \
                                          self._config.angle_up , dtype = tf.float32 )[0]

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

        clip_x1 = tf.maximum( 0. , clip_x1 )
        clip_y1 = tf.maximum( 0. , clip_y1 )
        clip_x2 = tf.minimum( clip_x2 , 1. )
        clip_y2 = tf.minimum( clip_y2 , 1. )

        clip_width = clip_x2 - clip_x1
        clip_height= clip_y2 - clip_y1

        offset_height = tf.to_int32 ( clip_y1 * height )
        offset_width  = tf.to_int32 ( clip_x1 * height )
        target_height = tf.to_int32 ( clip_height * height )
        target_width  = tf.to_int32 ( clip_width * width )

        landmarks = tf.concat( [ ( landmarks[0:5] - clip_x1 ) / clip_width , \
                                 ( landmarks[5:]  - clip_y1 ) / clip_height ] , axis = -1 )

        tf_image = tf.image.crop_to_bounding_box( tf_image , offset_height, \
                                                  offset_width , target_height , target_width )

        return tf_image , landmarks

    def _parser( self , line ):
        """
        data line in annotation file in a form:
          relative_data_path  x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 #gender #smile #glass #pose

        :return ( imgs , normalized_landmarks )

        """

        FIELD_DEFAULT = [ ['IMAGE_PATH'] , [0.], [0.],[0.],[0.],[0.],[0.],\
                [0.],[0.],[0.],[0.], [0], [0] , [0] , [0] ]

        fields = tf.decode_csv( line , FIELD_DEFAULT , field_delim = ' ' )
        content = tf.read_file( self._data_path + '/' + fields[0] )

        landmarks = tf.stack( [ fields[1], fields[2], fields[3], fields[4], \
                fields[5], fields[6], fields[7], fields[8], fields[9],\
                fields[10] ] , axis = -1 )

        tf_image = tf.image.decode_jpeg( content , channels = 3 )

        height = tf.to_float( tf.shape( tf_image )[0] )
        width  = tf.to_float( tf.shape( tf_image )[1] )

        transformed_landmarks = tf.concat( [ landmarks[0:5] / width, \
                landmarks[5:10] / height] , -1 )

        return tf_image , transformed_landmarks


    def _computeInnerBoundary( self , xmin, xmax, ymin, ymax ):
        bbox_width  = xmax - xmin
        bbox_heigth = ymax - ymin

        new_xmin = xmin - self._config.inner_left_adding * bbox_width
        new_xmax = xmax + self._config.inner_right_adding * bbox_width
        new_ymin = ymin - self._config.inner_up_adding * bbox_heigth
        new_ymax = ymax + self._config.inner_down_adding * bbox_heigth

        return new_xmin, new_xmax, new_ymin, new_ymax

    def _computeOuterBoundary( self , xmin , xmax , ymin , ymax ):
        bbox_width  = xmax - xmin
        bbox_heigth = ymax - ymin

        new_xmin = xmin - self._config.outer_left_adding * bbox_width
        new_xmax = xmax + self._config.outer_right_adding * bbox_width
        new_ymin = ymin - self._config.outer_up_adding * bbox_heigth
        new_ymax = ymax + self._config.outer_down_adding * bbox_heigth

        return new_xmin, new_xmax, new_ymin, new_ymax

if __name__ == "__main__":
   
    args.data_path = "/home/public/data"
    args.tempdir_path = "/home/public/data/tmp"
    args.anno_path = "/home/public/data/celebrity_lmk"
    args.if_augmentation = True
    args.train_batch_size = 100

    ms_data = MSCELEB( args )
    #ms_data = MTFL( args )

    sess = tf.InteractiveSession()
    ms_data.showBatch(  sess , ms_data.trainDataStream )
