from numpy import zeros,array,arange,max,min
from numpy.random import randint


def calc_output(input_size,kernel_size,padding_size,stride):
    return int(((input_size - kernel_size + 2*padding_size)/stride)+1)

def calc_input(out_size,kernel_size,padding_size,stride):
    return stride*(out_size-1)+kernel_size-2*padding_size

def calc_kernel_size(out_size,input_size,padding_size,stride):
    return -stride*(out_size-1)+input_size+2*padding_size

def get_grid_locations(img_dims,n_grids):

    grid_locations = {}

    div_size= img_dims[0]/n_grids

    for i in range(n_grids):
        for j in range(n_grids):

            grid_min_x = (0+i)*div_size
            grid_min_y = (0+j)*div_size

            grid_max_x = (1+i)*div_size
            grid_max_y = (1+j)*div_size

            grid_locations[(i,j)] = (grid_min_x,grid_min_y,grid_max_x,grid_max_y)

    return grid_locations

def find_centre(min_x,min_y,max_x,max_y):
    return (min_x+max_x)/2,(min_y+max_y)/2

def get_all_centers(boxes):
    min_xs, min_ys, max_xs, max_ys = boxes[...,0],boxes[...,1],boxes[...,2],boxes[...,3],
    return find_centre(min_xs, min_ys, max_xs, max_ys)

def get_wh(boxes):
    min_xs, min_ys, max_xs, max_ys = boxes[...,0],boxes[...,1],boxes[...,2],boxes[...,3],
    return (max_xs-min_xs),(max_ys-min_ys)

def get_bboxes(n_classes,norm_class_vector):

    """ converts a normal class label to a dict for easy reference """

    size = 5+n_classes
    n_boxes = int(len(norm_class_vector)/size)

    bboxes = {}

    for i in range(n_boxes):
        bboxes[i] = (norm_class_vector[size*i:size*(i+1)])

    return bboxes

def is_inside(min_x,min_y,max_x,max_y,cx,cy):
    if cx>min_x and cx<max_x and cy>min_y and cy<max_y:
        return True
    else:
        return False

def convert_bbox_to_output_tensor(img_dims, n_grids,n_classes,norm_class_vector):

    """
    this will convert a normal output vector of shape (n_samples, ((x, y, w, h, c) * nboxes) to

    a yolo label of shape (n_grids,n_grids,n_classes)

    works on only 1 sample

    """

    label_arr = zeros((n_grids,n_grids,5+n_classes))
    boxes = get_bboxes(n_classes,norm_class_vector)

    locs = get_grid_locations(img_dims,n_grids)

    for box in boxes:
        for loc in locs:
            cx,cy = find_centre(*boxes[box][0:4])
            if is_inside(*locs[loc],cx,cy):
                label_arr[loc] = boxes[box]

    return label_arr

def convert_int_to_one_hot(num,n_classes):
    temp = zeros(n_classes)
    temp[num] = 1
    return temp.tolist()

def convert_all_labels(image_dims,labels,n_classes,n_grids):
    num_samples = labels.shape[0]
    tensor = zeros((num_samples,n_grids,n_grids,5+n_classes))

    for i in range(num_samples):
        tensor[i] = convert_bbox_to_output_tensor(image_dims,n_grids,n_classes,labels[i])

    return tensor

def scale_input_label_tensor(img_dim,tensor,scale_down=True):
    n_classes = tensor.shape[3]-5
    n_samples = tensor.shape[0]
    scaled_tensor = tensor

    for i in range(n_samples):
        for j in range(n_classes):
            for k in range(n_classes):
                if scale_down:
                    # scale to between 0 and 1
                    scaled_tensor[i][j,k][0:4] = tensor[i][j,k][0:4]/img_dim[0]
                else:
                    # scale back to 0 and img_dim
                    scaled_tensor[i][j, k][0:4] = tensor[i][j, k][0:4]*img_dim[0]
    return scaled_tensor

def make_test_images(n_samples=1000,img_size=(28,28),n_boxes=1,n_classes=1):
    w,h = img_size

    boxes = zeros((n_samples,w,h,3))

    label_vect_size = 5+n_classes

    labels = zeros((n_samples,label_vect_size*n_boxes)) # n_samples, ((x, y, w, h, c) * nboxes)

    boxes[::] = [255,255,255] # make all images white

    class_color_int = int(255/n_classes)

    class_colors = [(255-c,255-int(c/2),c) for c in arange(0,255,class_color_int)]

    for j in range(n_boxes):
        for i in range(n_samples):

            # allow the starting point for a box to be within
            min_x = randint(0,w)
            min_y = randint(0,h)

            max_x = randint(min_x,w)
            max_y = randint(min_y,h)

            cls = randint(0,n_classes)
            color = class_colors[cls]
            boxes[i][min_x:max_x, min_y:max_y] = color



            box_index_start = (0+j)*label_vect_size
            # assign to spot in long vector
            labels[i][box_index_start:box_index_start+label_vect_size] = [min_x, min_y, max_x, max_y, 1]+convert_int_to_one_hot(cls,n_classes) # set labels

            # scale image x,y,w,h so that it is between 0 and 1
            # labels[i][box_index_start:box_index_start+4] = labels[i][box_index_start:box_index_start+4] / img_size[0]

    # assert labels.max() <= 1 and labels.min() >=0
    boxes = boxes/255 # scale so that images are between 0 and 1
    return (boxes,labels)

def test_model(img_dims,n_classes,n_boxes,model):
    import cv2

    tX,_ = make_test_images(n_samples=10,img_size=img_dims,n_boxes=n_boxes,n_classes=n_classes)
    test_img = randint(0,10)

    prediction = model.predict(array([tX[test_img]]))[0]
    rescaled_prediction = scale_input_label_tensor((128, 128), array([prediction]), False)[0]

    n_grids = prediction.shape[0]

    for i in range(n_grids):
        for j in range(n_grids):

            if rescaled_prediction[i,j][4]:
                x_min,y_min,x_max,y_max = rescaled_prediction[i,j][0:4]

                tX[test_img] = cv2.rectangle(tX[test_img],(x_min,y_min),(x_max,y_max),color=(0,0,0))

    # return prediction

    cv2.imshow('',tX[test_img])
    cv2.waitKey(1000)

def calc_iou(a_choords,b_choords):
    a_min_x,a_min_y,a_max_x,a_max_y=a_choords
    b_min_x,b_min_y,b_max_x,b_max_y=b_choords

    assert a_min_x < a_max_x
    assert a_min_y < a_max_y
    assert b_min_x < b_max_x
    assert b_min_y < b_max_y

    # determine the coordinates of the intersection rectangle
    x_left = max(a_min_x, b_min_x)
    y_top = max(a_min_y, b_min_y)
    x_right = min(a_max_x, b_max_x)
    y_bottom = min(a_max_y, b_max_y)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (a_max_x - a_min_x) * (a_max_y - a_min_y)
    bb2_area = (b_max_x - b_min_x) * (b_max_y - b_min_y)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0

    return iou



"""
N.B.: the structure below perfectly returns a 10x10x15

    model = Sequential()
    model.add(Conv2D(64, (7, 7), strides=1, input_shape=(512, 512, 3)))  # out (None, 506, 506, 64)
    model.add(Conv2D(64, (7, 7), strides=2, ))  # out (None, 250, 250, 64)
    model.add(MaxPool2D())  # out (None, 125, 125, 64)

    model.add(Conv2D(128, (5, 5), strides=2))  # out  (None, 61, 61, 128)
    model.add(Conv2D(128, (5, 5), strides=2, ))  # out (None, 29, 29, 128)
    model.add(MaxPool2D())  # out (None, 14, 14, 128)

    model.add(Conv2D(15, (5, 5), strides=1))  # out (None, 10, 10, 15)

"""


