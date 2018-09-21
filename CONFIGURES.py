
class args( object ):
    """
    args for the whole tcdcn algorithm
    including data preprocessing and augmentation
    and the whole CNN training

    if you wanna make an change, please do not modify this file
    just inherit this class
    """
    # data related
    data_path = "/Users/pitaloveu/working_data/MTFL"
    tempdir_path = "/Users/pitaloveu/working_data/MTFL/tmp"
    train_batch_size = 20
    test_batch_size = 5

    # data preprocessing parameters
    if_normalize = True

    if_shuffle = True
    if_bright = False
    if_contrast = False

    # here the augmentation will be done in three aspects:
    #   (1). translation in X and Y direction
    #   (2). rotation
    #   (3). zoom in or out
    if_augmentation = True

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
    outer_left_adding = 1.
    outer_right_adding = 1.
    outer_up_adding = 1.
    outer_down_adding = 1.

    # doing the clockwise rotation augmentation
    # the rotation angle will lay in [ 0, 10 ] degree
    angle_down = -180
    angle_up = 180
