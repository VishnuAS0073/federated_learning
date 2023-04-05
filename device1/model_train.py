
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import time 
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import pickle


# def train():
# 	cwd = os.getcwd()
# 	if os.path.isdir(cwd + '/local_model') == False:
# 		os.mkdir(cwd + '/local_model')
# 	object_name = "object_name"
# 	main_path = cwd + '/image_dataset_path'
# 	train_path = main_path+'/train'
# 	valid_path = main_path+'/valid'
# 	test_path = main_path+'/test'

# 	train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
# 		directory=train_path, target_size=(224,224), batch_size=10)
# 	valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
# 		directory=valid_path, target_size=(224,224), batch_size=10)
# 	test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
# 		directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)
# 	start = time.time()
# 	mobile = tf.keras.applications.mobilenet.MobileNet()
# 	x = mobile.layers[-6].output
# 	output = Dense(units=2, activation='softmax')(x)
# 	model = Model(inputs=mobile.input, outputs=output)

# 	for layer in model.layers[:-5]:
# 		layer.trainable = False

# 	model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 	history = model.fit(train_batches,
# 			  steps_per_epoch=len(train_batches),
# 			  validation_data=valid_batches,
# 			  validation_steps=len(valid_batches),
# 			  epochs=10,
# 			  verbose=1,use_multiprocessing = False
# 	)
# 	model.save(cwd + "/local_model/model1.h5")
# 	x = history.history
# 	end = time.time() - start
# 	return (x,object_name)


def train():
	cwd = os.getcwd()
	if os.path.isdir(cwd + '/local_model') == False:
		os.mkdir(cwd + '/local_model')
	object_name = "object_name"

	main_path = cwd+'\\Image_dataset_path'
	print(main_path)
	# Set the dimensions of the image
	img_width, img_height = 224, 224
	
	# Set the paths for the training, validation, and test data
	train_data_dir = main_path+'\\Train'
	validation_data_dir = main_path+'\\Valid'
	test_data_dir = main_path+'\\Test'
	print("file readed")
	# Set the number of epochs and batch size
	epochs = 5
	batch_size = 64
	
	# Create the CNN model
	model = Sequential()
	# model.add(Conv2D(input_shape = (32,32,3), filters = 8, kernel_size = (5,5),activation = "relu", padding = "same" ))
	model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(5, activation='softmax'))
	print("model created")
	print(model)
	# Compile the model
	# Prepare the training data and validation data using data augmentation
	train_datagen = ImageDataGenerator(rescale=1./255,
									shear_range=0.2,
									zoom_range=0.2,
									horizontal_flip=True)
	validation_datagen = ImageDataGenerator(rescale=1./255)
	test_datagen = ImageDataGenerator(rescale=1./255)
	train_generator = train_datagen.flow_from_directory(main_path,
														target_size=(img_width, img_height),
														batch_size=batch_size,
														class_mode='categorical',classes=["train"])
	print(train_generator.samples)
	validation_generator = validation_datagen.flow_from_directory(main_path,
																target_size=(img_width, img_height),
																batch_size=batch_size,
																class_mode='categorical',classes=["valid"])
	test_generator = test_datagen.flow_from_directory(main_path,
														target_size=(img_width, img_height),
														batch_size=batch_size,
														class_mode='categorical',classes=["test"])


	print("model compiled")
	# Define the callbacks

	early_stopping = EarlyStopping(monitor='val_loss', patience=5)
	model_checkpoint = ModelCheckpoint('diabetic_retinopathy_detection.h5', save_best_only=True)
	print("moled stop")
	print(train_generator.samples)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)
	print(model.summary())
	# Train the model
	# model.fit(train_generator,steps_per_epoch=train_generator.samples // batch_size,epochs=epochs,
	# validation_data=validation_generator,
	# validation_steps=validation_generator.samples // batch_size,
	# callbacks=[early_stopping, model_checkpoint])
	print("model trained")
	# Evaluate the model on the test data
	test_generator.reset()
	Y_pred = model.predict(test_generator, steps=test_generator.samples // 
	batch_size + 1)
	y_pred = np.argmax(Y_pred, axis=1)
	print('Confusion Matrix')
	print(confusion_matrix(test_generator.classes, y_pred))
	
	# Save the model
	model.save(cwd + "/local_model/model1.h5")
	# x = history.history
	# end = time.time() - start
	# return (x,object_name)
