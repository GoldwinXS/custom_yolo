from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Reshape
import tensorflow as tf
from keras import backend as K

from ProjectUtils import *


img_dims = 130, 130
n_grids = 10
n_classes = 2
n_boxes = 2

X, labels = make_test_images(n_samples=1000, img_size=img_dims, n_boxes=n_boxes, n_classes=10)
y = convert_all_labels(img_dims, labels, 10, 10)
y = scale_input_label_tensor(img_dims, y)

@tf.function
def yolo_loss(y_pred,y_true):

    pred_boxes = y_pred[...,:4]
    true_boxes = y_true[...,:4]

    # use boxes to get proper xy and wh as from the paper
    pred_cx,pred_cy = get_all_centers(pred_boxes)
    true_cx,true_cy = get_all_centers(true_boxes)

    pred_w,pred_h = get_wh(pred_boxes)
    true_w,true_h = get_wh(true_boxes)

    # create masks
    pred_confs = y_pred[...,4]
    true_confs = y_true[...,4] # this is equivalent to 1obj from the paper... it will be 1 if there is supposed to be something there


    obj = true_confs.__copy__()
    noobj = tf.where(obj==0,1,0)
    noobj = tf.cast(noobj,tf.float32)

    pred_clss  = y_pred[...,5:]
    true_clss  = y_true[...,5:]

    xy_loss = K.sum(obj*(
            ((pred_cx-true_cx)**2)
            +
            ((pred_cy-true_cy)**2)))

    wh_loss = K.sum(obj*(((pred_w-true_w)**2)+
                         ((pred_h-true_h)**2)))

    loc_loss = xy_loss+wh_loss

    conf_loss =K.sum(obj*(pred_confs-true_confs)**2)\
               +\
               K.sum(noobj*(pred_confs-true_confs)**2)


    cls_loss = K.sum(obj*(K.sum((pred_clss-true_clss)**2,axis=-1)))

    loss = loc_loss+conf_loss+cls_loss


    tf.print(xy_loss)
    tf.print(wh_loss)
    tf.print(conf_loss)
    tf.print(cls_loss)

    return loss

def make_yolo(image_dims):
    model = Sequential()
    model.add(Conv2D(64, (7, 7), strides=1, input_shape=image_dims + (3,)))  # out (None, 506, 506, 64)
    model.add(Conv2D(64, (7, 7), strides=2, ))  # out (None, 250, 250, 64)
    model.add(MaxPool2D())  # out (None, 125, 125, 64)

    model.add(Conv2D(128, (5, 5), strides=2))  # out  (None, 61, 61, 128)
    model.add(Conv2D(128, (5, 5), strides=2, ))  # out (None, 29, 29, 128)
    model.add(MaxPool2D())  # out (None, 14, 14, 128)

    model.add(Conv2D(15, (5, 5), strides=1))  # out (None, 10, 10, 15)

    model.add(Flatten())

    model.add(Dense(10 * 10 * 15))
    # model.add(Dense(10*10*15,activation='sigmoid'))
    model.add(Reshape((10, 10, 15), input_shape=(10 * 10 * 15,)))

    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model

def make_yolo_small(image_dims):
    model = Sequential()
    model.add(Conv2D(64, (7, 7), strides=2, input_shape=image_dims + (3,)))  # out (None, 506, 506, 64)
    model.add(Conv2D(64, (7, 7), strides=2, ))  # out (None, 250, 250, 64)
    model.add(MaxPool2D())  # out (None, 125, 125, 64)

    model.add(Conv2D(64, (5, 5), strides=1))  # out (None, 506, 506, 64)
    model.add(Conv2D(64, (5, 5), strides=1))  # out (None, 250, 250, 64)
    model.add(MaxPool2D())  # out (None, 125, 125, 64)

    model.add(Flatten())

    # model.add(Dense(10 * 10 * 15))
    model.add(Dense(10*10*15,activation='sigmoid'))
    model.add(Reshape((10, 10, 15), input_shape=(10 * 10 * 15,)))



    model.compile(optimizer='Adam', loss=yolo_loss)
    model.summary()

    return model


model = make_yolo_small(img_dims)
model.load_weights('yolo_test.h5')

model.fit(X, y, epochs=1,batch_size=128)
# model.save_weights('yolo_test.h5')

model.load_weights('yolo_test.h5')
p = model.predict(X)
yolo_loss(K.constant(p),K.constant(y))


for i in range(5):
    test_model(img_dims, n_classes, n_boxes, model)
