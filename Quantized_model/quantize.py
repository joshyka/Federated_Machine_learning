# -*- coding: utf-8 -*-
"""Quantize.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kAEguSRzrKxdCpDKuUvEmeMTB9MRU6XX
"""

pip install -q tensorflow-model-optimization

from keras.models import load_model
model=load_model('global_model_100.h5')
model.summary()

import tensorflow_model_optimization as tfmot
from tensorflow import keras

quantize_model = tfmot.quantization.keras.quantize_model
# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)
optimizer = keras.optimizers.Adam(lr=0.0001)
# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer=optimizer,loss='mean_squared_error')
q_aware_model.summary()

q_aware_model.save('Intel_quantize_aware_model.h5')

# TensorFlow Lite converts full floating point to 8-bit integers
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
with open('Intel_QAT.tflite', 'wb') as f:
  f.write(quantized_tflite_model)

# TensorFlow Lite converts full floating point to half-precision floats (float16).
# import tensorflow as tf
# optimize="Speed"
# if optimize=='Speed':
#     converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
# elif optimize=='Storage':
#      converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# else:    
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
# #reduce the size of a floating point model by quantizing the weights to float16
# converter.target_spec.supported_types = [tf.float16]
# #save the quanitized model toa binary file
# quantized_tflite_model = converter.convert()
# with open('Intel_QAT.tflite16', 'wb') as f:
#   f.write(quantized_tflite_model)

