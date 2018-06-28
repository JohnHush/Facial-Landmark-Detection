# implement Facial Landmark Detection of TCDCN
# using tensorflow custom Estimator and Dataset API
# written by John Hush in Chengdu
# 06/19/2018
# cft

import tensorflow as tf
import try_fetch

# add model to do the test

def model( features , labels , mode , params ):
    # in the feature columns, user should provide features including
    # images and labels
    # the transfer will be done by tf automatically, e.g one-hot transformation
    # images = tf.feature_column.input_layer(features, params['feature_columns'])

    # directly use features and labels dict

    #input = tf.feature_column.input_layer( features , params['feature_columns'] )
    input = features['image']
    yy = features['smile']

    xx = tf.Print( yy , [yy,] , "smile value" )

    for units in params['hidden_units']:
        input = tf.layers.dense( input , units = units , activation = tf.nn.relu )

    # compute logits
    logits = tf.layers.dense( input , params['n_classes'] , activation=None)

    predicted_class = tf.argmax( logits , 1 )

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids' : predicted_class[:, tf.newaixs],
            'probabilities' : tf.nn.softmax( logits ),
            'logits' : logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=xx, logits=logits)

    accuracy = tf.metrics.accuracy(labels=xx,
                                   predictions=predicted_classes,
                                   name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == "__main__":

    my_feature_column = []

    my_feature_column.append( tf.feature_column.numeric_column(key = 'image', shape = [100] ) )

    classifier = tf.estimator.DNNClassifier( feature_columns= my_feature_column,
                                             hidden_units= [10,10] , n_classes= 2)

    classifier.train( input_fn= lambda : try_fetch.train_eval_input_fn( "/Users/pitaloveu/working_data/MTFL",
                                                               if_train = True, batch_size= 128 ),
                      steps = 100 )

    evaluate_result = classifier.evaluate( input_fn= lambda : try_fetch.train_eval_input_fn( "/Users/pitaloveu/working_data/MTFL",
                                                                                             if_train = False, batch_size = 128 ))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**evaluate_result))
    # make a regressor
