# example of preparing the horses and zebra dataset
import numpy as np
from os import listdir
from numpy import asarray, vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

# load all images in a directory into memory

def load_images(path, size=(256,256)):
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return asarray(data_list)

def select_fraction(data, fraction=1):
	nrOfItems = int(data.shape[0] * min(fraction, 1))
	indices = np.arange(data.shape[0])
	# choose which images to use in the set
	indices = np.random.choice(indices, nrOfItems, replace=False)
	# return only the randomly selected images
	return np.array(data)[indices]


# dataset path
path = '../Unmodified Dataset/'
# fraction of the dataset to use
trainingFraction = 1
testFraction = 1
# load dataset A
dataA1 = select_fraction(load_images(path + 'trainA/'), trainingFraction)
dataAB = select_fraction(load_images(path + 'testA/'), testFraction)
dataA = vstack((dataA1, dataAB))
print('Loaded dataA: ', dataA.shape)
# load dataset B
dataB1 = select_fraction(load_images(path + 'trainB/'), trainingFraction)
dataB2 = select_fraction(load_images(path + 'testB/'), testFraction)
dataB = vstack((dataB1, dataB2))
print('Loaded dataB: ', dataB.shape)
# save as compressed numpy array
filename = 'summer2winter_256.npz'
savez_compressed(filename, dataA, dataB)
print('Saved dataset: ', filename)