from flask import Flask, render_template, request
import os
import shutil
import cv2
import math
from PIL import Image as img
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import sequence
from keras.models import Model

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'upload')
INPUT_FRAME_FOLDER = os.path.join(UPLOAD_FOLDER, 'input_frames')
MODEL_FOLDER = os.path.join(APP_ROOT, 'static', 'trained_model')
CNN_MODEL_FILENAME = 'cnn_rnn_model.h5'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def upload_file():
	return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_files():
	predict = [0] * 6
	if request.method == 'POST':
		f = request.files['file']
		if os.path.exists(app.config['UPLOAD_FOLDER']) and os.path.isdir(app.config['UPLOAD_FOLDER']):
			shutil.rmtree(app.config['UPLOAD_FOLDER'])
		os.mkdir(app.config['UPLOAD_FOLDER'])
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], "input.mp4"))
		images = preprocess_data()
		predict = get_prediction(images)

	return render_template('result.html', predict=predict)


def preprocess_data():
	images = [[]]
	#generating frames using opencv
	video_path = os.path.join(app.config['UPLOAD_FOLDER'], "input.mp4")
	if os.path.exists(video_path):
		cap = cv2.VideoCapture(video_path)
		frameRate = cap.get(cv2.CAP_PROP_FPS)
		count = 0
		os.mkdir(INPUT_FRAME_FOLDER)
		while (cap.isOpened()):
			frameId = cap.get(cv2.CAP_PROP_POS_FRAMES)
			ret, frame = cap.read()
			if (ret != True):
				break
			if (frameId % math.floor(frameRate) == 0) or (
					frameId % math.floor(frameRate) == (math.floor(frameRate) // 2)):
				filename = "frame%d.jpg" % count
				count += 1
				h, w, c = frame.shape
				y = (w - h) // 2
				good = frame[:, y:y + h]
				good = cv2.resize(good, (224, 224))
				cv2.imwrite(os.path.join(INPUT_FRAME_FOLDER, filename), good)
				image = img.open(os.path.join(INPUT_FRAME_FOLDER, filename))
				images[0].append(np.asarray(image))
		cap.release()
		images = np.asarray(images)
		# print(images[0].shape)
		# for i in range(len(images[0])):
		# 	plt.figure()
		# 	plt.imshow(images[0][i])
		# 	plt.show()
	return images


def get_prediction(images):
	# vgg16_cnn_images = []
	# model = keras.models.load_model(os.path.join(MODEL_FOLDER, CNN_MODEL_FILENAME))
	# vgg16_model = VGG16(weights='imagenet')
	# vgg16_extractor = Model(inputs=vgg16_model.input, outputs=vgg16_model.get_layer('fc2').output)
	# vgg16_cnn_images.append(vgg16_extractor.predict(images[0]))
	# vgg16_cnn_images = np.asarray(vgg16_cnn_images)
	#
	# print(vgg16_cnn_images.shape)
	#
	# input = sequence.pad_sequences(vgg16_cnn_images, maxlen=15)
	# print("Vinh", input.shape)
	# predict = model.predict(input[0])
	# print(predict)
	predict = [0.12, 0.23, 0.34, 0.45, 0.51, 0.62]
	return [int(p*100) for p in predict]

if __name__ == '__main__':
   app.run(debug = True)