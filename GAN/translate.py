import tensorflow_addons as tfa
import numpy as np

from numpy import load
from numpy.random import randint
from matplotlib import pyplot
from keras.models import Input
from tensorflow.keras.models import load_model
from tensorflow_addons.layers.normalizations import InstanceNormalization

#from train import InstanceNormalization

def load_models():
    ganAtoB = load_model('ganmodel/g_model_AtoB_007700.h5', compile=False)
    ganBtoA = load_model('ganmodel/g_model_BtoA_007700.h5', compile=False)
    vitAtoB = load_model('vitmodel/g_model_AtoB_007700.h5', compile=False)
    vitBtoA = load_model('vitmodel/g_model_BtoA_007700.h5', compile=False)
    return (ganAtoB, ganBtoA, vitAtoB, vitBtoA)

def load_samples(filename):
	# load the dataset
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

def predict_image(model, image):
	# generate fake instance
	X = model.predict(image)
	return X

def select_real_images(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	return X

def generate_fake_images(g_model, dataset):
    # generate fake instance
    X = g_model.predict(dataset)
    return X

def translate():
    (ganAtoB, ganBtoA, vitAtoB, vitBtoA) = load_models()
    A, B = load_samples('summer2winter_256.npz')
    n_samples = 5
    original = select_real_images(A, n_samples)

    ganGenerated = generate_fake_images(ganAtoB, original)
    vitGenerated = generate_fake_images(vitAtoB, original)

    original = (original + 1) / 2.0
    ganGenerated = (ganGenerated + 1) / 2.0
    vitGenerated = (vitGenerated + 1) / 2.0

    pyplot.figure(figsize=(10,10))
    for i in range(n_samples):
        pyplot.subplot(n_samples, 3, 3*i+1)
        pyplot.axis('off')
        pyplot.imshow(original[i])
        pyplot.subplot(n_samples, 3, 3*i+2)
        pyplot.axis('off')
        pyplot.imshow(ganGenerated[i])
        pyplot.subplot(n_samples, 3, 3*i+3)
        pyplot.axis('off')
        pyplot.imshow(vitGenerated[i])
    pyplot.savefig("comparison")

translate()
