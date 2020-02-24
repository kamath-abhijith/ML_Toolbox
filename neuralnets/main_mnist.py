import os
import struct

import numpy as np

from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

## Read mnist data
train_filename = '/Users/abijithjkamath/Desktop/TECHNOLOGIE/RawData/mnist/train-images-idx3-ubyte'
with open(train_filename, 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    mnist_train = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

test_filename = '/Users/abijithjkamath/Desktop/TECHNOLOGIE/RawData/mnist/t10k-images-idx3-ubyte'
with open(test_filename, 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    mnist_test = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

labels_filename = '/Users/abijithjkamath/Desktop/TECHNOLOGIE/RawData/mnist/train-labels-idx1-ubyte'
with open(labels_filename, 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    mnist_train_labels = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

test_labels_filename = '/Users/abijithjkamath/Desktop/TECHNOLOGIE/RawData/mnist/t10k-labels-idx1-ubyte'
with open(test_labels_filename, 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    mnist_test_labels = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

## Preprocessing
[num_test_samples, m,n] = mnist_train.shape
test_data = mnist_train.reshape(num_test_samples,m*n)/255.0

[num_train_samples, m,n] = mnist_test.shape
train_data = mnist_test.reshape(num_train_samples,m*n)/255.0

lb = LabelBinarizer()
mnist_train_labels = lb.fit_transform(mnist_train_labels)
mnist_test_labels = lb.transform(mnist_test_labels)

## Define model
model = Sequential()
model.add(Dense(512, input_shape=(784,1), activation="sigmoid"))
model.add(Dense(256, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

## Initialise training
INIT_LR = 1e-2
EPOCHS = 10

print("[MNIST] Training Network ... ")

opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

## Train the network
H = model.fit(train_data, mnist_train_labels, validation_data=(test_data,mnist_test_labels), epochs=EPOCHS, batch_size=32)

## Validation
print("[MNIST] Validating ... ")
predictions = model.predict(test_data, batch_size=32)
print(classification_report(mnist_test_labels.argmax(axis=1),predictions.argmax(axis=1),target_names=lb.classes_))

## Plots
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()