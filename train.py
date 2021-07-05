# import the necessary packages
from config import config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from imutils import paths

# load the contents of the CSV annotations file
print("[INFO] loading dataset...")
# rows = open(config.ANNOTS_PATH).read().strip().split("\n")
# initialize the list of data (images), our target output predictions
# (bounding box coordinates), along with the filenames of the
# individual images
data = []
labels = []
targets = []
filenames = []

IMG_SIZE = 256

# loop over all CSV files in the annotations directory
for csvPath in paths.list_files(config.ANNOTS_PATH, validExts=(".csv")):
	# load the contents of the current CSV annotations file
	rows = open(csvPath).read().strip().split("\n")

	# loop over the rows
	for row in rows:
		# break the row into the filename, bounding box coordinates,
		# and class label
		row = row.split(",")
		(filename, w, h, label, startX, startY, endX, endY) = row
		# derive the path to the input image, load the image (in
		# OpenCV format), and grab its dimensions
		imagePath = os.path.sep.join([config.IMAGES_PATH, label,
			filename])
		image = cv2.imread(imagePath)
		# print(imagePath)
		# (h, w) = image.shape[:2]

		# print(w)
		# print(type(int(w)))
		# scale the bounding box coordinates relative to the spatial
		# dimensions of the input image
		startX = float(startX) / int(w)
		startY = float(startY) / int(h)
		endX = float(endX) / int(w)
		endY = float(endY) / int(h)

		# load the image and preprocess it
		image = load_img(imagePath, target_size=(IMG_SIZE, IMG_SIZE))
		image = img_to_array(image)

		# update our list of data, class labels, bounding boxes, and
		# image paths
		data.append(image)
		labels.append(label)
		targets.append((startX, startY, endX, endY))
		filenames.append(imagePath)

# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")
# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
split = train_test_split(data, targets, filenames, test_size=0.10,
	random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]
# write the testing filenames to disk so that we can use then
# when evaluating/testing our bounding box regressor
print("[INFO] saving testing filenames...")
f = open(config.TEST_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()

#ARCHITECTURE FROM https://doi.org/10.1007/s12145-019-00434-8

print("[INFO] building architecture cnn...")
head = keras.Sequential()
#INPUT
head.add(keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))) 
#CONVOLUTIONAL HIDDEN LAYERS
head.add(layers.Conv2D(16, 3, activation="relu"))
head.add(layers.Conv2D(32, 3, activation="relu"))
head.add(layers.MaxPooling2D(2))
head.add(layers.Conv2D(32, 3, activation="relu"))
head.add(layers.MaxPooling2D(2))
head.add(layers.Conv2D(32, 3, activation="relu"))
head.add(layers.MaxPooling2D(2))
head.add(layers.Conv2D(32, 3, activation="relu"))
head.add(layers.MaxPooling2D(2))

head.add(layers.Dropout(0.5))
head.add(layers.Flatten())

#FULLY CONNECTED LAYERS
head.add(layers.Dense(6272, activation="relu"))
head.add(layers.Dropout(0.5))
head.add(layers.Dense(768, activation="relu"))
head.add(layers.Dropout(0.5))
head.add(layers.Dense(32, activation="relu"))

flatten = head.output
flatten = Flatten()(flatten)

#OUTPUT LAYER
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = Model(inputs=head.input, outputs=bboxHead)

# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(lr=config.INIT_LR)
model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])
print(model.summary())
# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS,
	verbose=1)
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")
# plot the model training history
N = config.NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)