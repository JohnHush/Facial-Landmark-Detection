import cv2
import tensorflow as tf
import numpy as np
from os.path import isfile, join

datapath = "/Users/pitaloveu/working_data/MTFL"
training = "training.txt"

#def wrapped_func( s ):

def LineFilter( line ):
    line = tf.py_func( lambda s: s.strip() , [line] , tf.string )
    line = tf.py_func( lambda s: s.split()[0] , [line]  , tf.string )
    #line = tf.py_func( lambda s: s.replace( "\\", "/" ) , [line] , tf.string )


    #return tf.py_func( lambda s: isfile( tf.string_join( [ tf.constant(datapath) , s] ) ) , [line] , tf.bool )
    return True


def ParseLine( line ):
    line = tf.py_func( lambda s: s.strip() , [line] , tf.string )
    defaults_format = [ [""] , [0.] , [0.] , [0.] , [0.] , 
            [0.], [0.], [0.], [0.], [0.], [0.], [0], [0], [0], [0] ]

    parse_line = tf.decode_csv( line , defaults_format , field_delim = " " )

    return parse_line[0]

dataset = tf.data.TextLineDataset( join( datapath , training ) )
#dataset = dataset.filter( LineFilter )
dataset = dataset.map( ParseLine )

fetch_one = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    print( sess.run( fetch_one ) )
    print( sess.run( fetch_one ) )
    print( sess.run( fetch_one ) )

