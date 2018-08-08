import numpy as np
from functools import reduce
import fetchData
import cv2
import os

data_path = "/Users/pitaloveu/working_data/MTFL"

def run_MSCELEB():
    """
    the workflow will be:
    1. load the information of all the labeled data

    2. for every single element in the query sequence, find the one
       in the labeled data

    3. normalize the landmark first, then compute the pixel error

    4. Loop all the qualified query one in, and average them
    """
    anno_path = "/home/public/data/celebrity_lmk"


    image_path , landmarks, gender, smile, glasses, pose = \
            fetchData.load_path( data_path , if_train = False )

    imdb = list( zip( image_path , landmarks ) )

    with open( 'mtcnn_test_results.txt' , 'r' ) as f:
        lines = f.readlines()

    lines = [ line for line in lines if \
            line.split( '|' )[-1].strip() != "" and \
            len( line.split( '|' )[-1].strip().split( ' ' ) ) == 10 ]

    errors = []
    for line in lines:
        line_split = list( map ( lambda s: s.strip() , line.split( '|' ) ))
        _key = line_split[0].split( '/' )[-1]
        predict_landmarks = list( map( lambda s: float( s ) , line_split[-1].split( ' ' ) ) )

        imdb_info = [ s for s in imdb if s[0].find(_key) != -1 ]
        assert len( imdb_info ) == 1

        img = cv2.imread( imdb_info[0][0] )
        labeled_landmarks = imdb_info[0][1]
        height = img.shape[0]
        width  = img.shape[1]

        # transfer the landmark labeled to ( 96, 112 ) format
        labeled_landmarks[0:5] = labeled_landmarks[0:5] * 96. / width
        labeled_landmarks[5:10] = labeled_landmarks[5:10] * 112. / height

        #error in a form [ x1, x2 , x3, x4, x5, y1, y2, y3, y4, y5 ]
        error_landmarks = np.array( predict_landmarks ) - labeled_landmarks
        error = np.sqrt( np.square( error_landmarks[0:5]) + \
                np.square( error_landmarks[5:10]) )

        errors.append( error )

    errors = np.array( errors )
    errors_mean = np.mean( errors , 0 )
    print( errors_mean )

def run():
    """
    the workflow will be:
    1. load the information of all the labeled data

    2. for every single element in the query sequence, find the one
       in the labeled data

    3. normalize the landmark first, then compute the pixel error

    4. Loop all the qualified query one in, and average them
    """
    image_path , landmarks, gender, smile, glasses, pose = \
            fetchData.load_path( data_path , if_train = False )

    imdb = list( zip( image_path , landmarks ) )

    with open( 'mtcnn_test_results.txt' , 'r' ) as f:
        lines = f.readlines()

    lines = [ line for line in lines if \
            line.split( '|' )[-1].strip() != "" and \
            len( line.split( '|' )[-1].strip().split( ' ' ) ) == 10 ]

    errors = []
    for line in lines:
        line_split = list( map ( lambda s: s.strip() , line.split( '|' ) ))
        _key = line_split[0].split( '/' )[-1]
        predict_landmarks = list( map( lambda s: float( s ) , line_split[-1].split( ' ' ) ) )

        imdb_info = [ s for s in imdb if s[0].find(_key) != -1 ]
        assert len( imdb_info ) == 1

        img = cv2.imread( imdb_info[0][0] )
        labeled_landmarks = imdb_info[0][1]
        height = img.shape[0]
        width  = img.shape[1]

        # transfer the landmark labeled to ( 96, 112 ) format
        labeled_landmarks[0:5] = labeled_landmarks[0:5] * 96. / width
        labeled_landmarks[5:10] = labeled_landmarks[5:10] * 112. / height

        #error in a form [ x1, x2 , x3, x4, x5, y1, y2, y3, y4, y5 ]
        error_landmarks = np.array( predict_landmarks ) - labeled_landmarks
        error = np.sqrt( np.square( error_landmarks[0:5]) + \
                np.square( error_landmarks[5:10]) )

        errors.append( error )

    errors = np.array( errors )
    errors_mean = np.mean( errors , 0 )
    print( errors_mean )

if __name__ == "__main__":
    run()
