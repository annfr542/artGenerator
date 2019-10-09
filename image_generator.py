#!/bin/python

# Values:
# OCTAVE_SCALE = 1.6
# base_model = "model3.h5"

from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

import tensorflow as tf
import numpy as np
import matplotlib as mpl

from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image

layer_dict = { }
layer_dict["model3.h5"] = ['conv2d_18', 'conv2d_19']
layer_dict["model_places.h5"] = ['block4_pool', 'block5_pool']
layer_dict["model_hybrid.h5"] = ['block4_pool', 'block5_pool']
layer_dict["model_places_flower_simple.h5"] = ['block5_pool', 'max_pooling2d_1']

def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)

def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return  tf.reduce_sum(losses)

@tf.function
def deepdream(model, img, step_size):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calc_loss(img, model)

    gradients = tape.gradient(loss, img)

    gradients /= tf.math.reduce_std(gradients) + 1e-8 

    img = img + gradients*step_size
    img = tf.clip_by_value(img, -1, 1)

    return loss, img

def run_deep_dream_simple(model, img, gui, steps, step_size, epoch, tot_epochs):
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    for step in range(steps):
        loss, img = deepdream(model, img, step_size)

        gui.set_output("Running: Epoch {}/{} Step {}, loss {}".format(epoch+1, tot_epochs, step+1, loss))
        gui.show_numpy_image(deprocess(img).numpy())

    result = deprocess(img)
    gui.show_numpy_image(result.numpy())

    return result

def load_model(model_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = script_dir + '/' + model_name
    base_model = tf.keras.models.load_model(path)

    return base_model

def load_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=[255,255,])
    img = np.array(img)
    return img

def generate(iters, step, scale, model, image, gui, epochs):
    base_model = load_model(model)
    original_img = load_image(image)

    names = layer_dict[model]
    layers = [base_model.get_layer(name).output for name in names]

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    img = tf.constant(np.array(original_img))
    base_shape = tf.cast(tf.shape(img)[:-1], tf.float32)

    for n in range(epochs):
        new_shape = tf.cast(base_shape*(scale**n), tf.int32)
        img = tf.image.resize(img, new_shape).numpy()
        img = run_deep_dream_simple(model=dream_model, img=img, gui=gui, steps=iters, step_size=step, epoch=n, tot_epochs=epochs)

    gui.set_output("Done")
    gui.show_numpy_image(img.numpy())
