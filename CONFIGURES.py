
class args( object ):
    """
    args for the whole tcdcn algorithm
    including data preprocessing and augmentation
    and the whole CNN training

    if you wanna make an change, please do not modify this file
    just inherit this class
    """
    # data related
    train_batch_size = 20
    if_shuffle = True

    # output data image size
    img_height = 224
    img_width  = 224

    #inner boundary, ratios w.r.t width and height of
    # the bbox of 5 landmarks
    inner_left_adding = 0.03
    inner_right_adding = 0.03
    inner_up_adding = 0.03
    inner_down_adding = 0.03

    # outer boundary, ratios w.r.t width and height of
    # the bbox of 5 landmarks
    outer_left_adding = 0.03
    outer_right_adding = 0.03
    outer_up_adding = 0.04
    outer_down_adding = 0.04

    # doing the clockwise rotation augmentation
    # the rotation angle will lay in [ 0, 10 ] degree
    angle_down = -15
    angle_up = 15
