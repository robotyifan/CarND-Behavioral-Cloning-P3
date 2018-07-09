import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt


lines = []
with open('/Users/Yifan/Desktop/CarND/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)


images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '/Users/Yifan/Desktop/CarND/CarND-Behavioral-Cloning-P3/data/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)



X_train = np.array(images)
y_train = np.array(measurements)

'''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Activation

model = Sequential()
model.add(Lambda(lambda x: x/ 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping= ((70,25),(0,0))))
model.add(Convolution2D(3,5,5))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss = 'mse' , optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch= 3)

model.save('model.h5')
'''

from alexnet import AlexNet
import tensorflow as tf
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

X_train, X_val = train_test_split(X_train, test_size = 0.2, random_state = 0)
y_train, y_val = train_test_split(y_train, test_size = 0.2, random_state = 0)

epochs = 1
nb_classes = 1
batch_size = 256

features = tf.placeholder(tf.float32, (None, 160,320,3))
labels = tf.placeholder(tf.int64, None)
resize = tf.image.resize_images(features, (227,227))

fc7 = AlexNet(resize, feature_extract = True)

shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8w = tf.Variable(tf.truncated_normal(shape, stddev = 1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8w, fc8b)

mse = logits - tf.cast(labels, tf.float32)
loss_op = tf.reduce_mean(tf.square(mse))
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op, var_list = [fc8w, fc8b])
init_op = tf.global_variables_initializer()

preds = tf.arg_max(logits, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

def eval_on_data(X, y ,sess):
	total_acc = 0
	total_loss = 0
	for offset in range(0, X.shape[0], batch_size):
		end = offset + batch_size
		X_batch = X[offset:end]
		y_batch = y[offset:end]

		loss, acc = sess.run([loss_op, accuracy_op], feed_dict = {features: X_batch, labels : y_batch})
		total_loss += (loss * X_batch.shape[0])
		total_acc += (acc * X_batch.shape[0])


	return total_loss/X.shape[0], total_acc/X.shape[0]

with tf.Session() as sess:
	sess.run(init_op)

	for i in range(epochs):
		X_train, y_train = shuffle(X_train, y_train)
		t0 = time.time()
		for offset in range(0,X_train.shape[0], batch_size):
			end = offset + batch_size
			sess.run(train_op,feed_dict = {features: X_train[offset:end], labels: y_train[offset:end]})

		val_loss, val_acc = eval_on_data(X_val, y_val, sess)
		print("Epoch", i+1)
		print("Time: %.3f seconds" % (time.time() - t0))
		print("Validation Loss = ", val_loss)
		print("Validation Accuracy = ", val_acc)
		print("")










































