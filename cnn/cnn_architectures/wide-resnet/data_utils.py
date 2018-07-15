
"""Contains utilities for data augmentation """

import math as m
import tensorflow as tf
import numpy as np

WIDTH = 32
HEIGHT = 32
DEPTH = 3
def _crop_random(image):
    """Randomly crops image and mask"""
    
    cond_crop = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    
    image = tf.cond(cond_crop, lambda: tf.image.resize_images(
        tf.random_crop(image, [int(HEIGHT * 0.85), int(WIDTH * 0.85), 3]),
        size=[HEIGHT, WIDTH]), lambda: tf.cast(image, tf.float32))
    
    return image

def _flip_random(image):
    """Randomly flips image left and right"""
    
    image = tf.image.random_flip_left_right(image)    
    image = tf.image.random_flip_up_down(image)

    return image

def _rotate_random(image):
    """Randomly rotate the image"""
    
    cond_rotate = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)

    div = tf.random_uniform([], minval=1,maxval=5, dtype=tf.int32)
    radian = tf.constant(m.pi) / tf.cast(div, tf.float32)

    image = tf.cond(cond_rotate,lambda:tf.cast(tf.contrib.image.rotate(image, radian), tf.float32),
                    lambda: tf.cast(image, tf.float32))
    
    return image

def _normalize_data(image):
    """Normalize image to zero mean and unit variance."""
    mean = tf.constant([125.3, 123.0, 113.9])
    var = tf.constant([63., 62.1, 66.7])
    image = tf.div(image -mean, var)
    return image

def data_augmentation(image, label):
    """Function that does random crop, flip, rotate"""
    image = _crop_random(image)
    image = _flip_random(image)
    image = _rotate_random(image)
                   
    return image, label

def _parse_function(serialized_example):
    
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)
    image = _normalize_data(image)
    
    return image, label

def input_pipeline(filename, batch_size,validation=False):
    """ Input data pipeline, no augmentation during validation"""

    with tf.device('/cpu:0'):
        # Read from tfrecords
        dataset = tf.data.TFRecordDataset(filename)
        if validation == False:
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.map(_parse_function, num_parallel_calls=6)

        # shuffle for only train set
        if validation == False:
            # here iam combining both normal and augmented samples
            augmented = dataset.map(data_augmentation, num_parallel_calls=6)            
            dataset = dataset.concatenate(augmented)
            dataset = dataset.shuffle(buffer_size=20000)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=10000)

        return dataset

