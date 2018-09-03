
"""Contains utilities for data augmentation """

import math as m
import tensorflow as tf
import numpy as np

INPUT_WIDTH = 128
INPUT_HEIGHT = 128

def _crop_random(image, mask):
    """Randomly crops image and mask"""
    seed = np.random.randint(42)
    
    cond_crop = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)
    
    image = tf.cond(cond_crop, lambda: tf.image.resize_images(
        tf.random_crop(image, [int(INPUT_HEIGHT * 0.75), int(INPUT_WIDTH * 0.75), 3], seed=seed),
        size=[INPUT_HEIGHT, INPUT_WIDTH]), lambda: tf.cast(image, tf.float32))
    mask = tf.cond(cond_crop, lambda: tf.image.resize_images(
        tf.random_crop(mask, [int(INPUT_HEIGHT * 0.75), int(INPUT_WIDTH * 0.75), 1], seed=seed), 
        size=[INPUT_HEIGHT, INPUT_WIDTH]),lambda: tf.cast(mask, tf.float32))
    
    return image, mask

def _flip_random(image, mask):
    """Randomly flips image left and right"""
    
    seed = np.random.randint(42)
    image = tf.image.random_flip_left_right(image, seed=seed)
    mask = tf.image.random_flip_left_right(mask, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    mask = tf.image.random_flip_up_down(mask, seed=seed)

    return image, mask

def _rotate_random(image, mask):
    """Randomly rotate the image"""
    
    cond_rotate = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)

    div = tf.random_uniform([], minval=1,maxval=5, dtype=tf.int32)
    radian = tf.constant(m.pi) / tf.cast(div, tf.float32)

    image = tf.cond(cond_rotate,lambda:tf.cast(tf.contrib.image.rotate(image, radian), tf.float32),
                    lambda: tf.cast(image, tf.float32))
    mask = tf.cond(cond_rotate,lambda:tf.cast(tf.contrib.image.rotate(mask, radian), tf.float32),
                    lambda: tf.cast(mask, tf.float32))                   

    return image, mask

def _normalize_data(image, mask):
    """Normalize image and mask within range 0-1."""
    image = image / 255.0
    mask = mask / 255.0

    return image, mask

def data_augmentation(image, mask):
    
    image, mask = _crop_random(image, mask)
    image, mask = _flip_random(image, mask)
    image, mask = _rotate_random(image, mask)
                   
    return image, mask

