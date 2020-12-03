from flask import Flask, render_template, request, make_response
import os
import shutil
import cv2
import math
from PIL import Image as img
import numpy as np
from tensorflow import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import sequence
from keras.models import Model
import time
# import matplotlib.pyplot as plt

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'upload')
INPUT_FRAME_FOLDER = os.path.join(UPLOAD_FOLDER, 'input_frames')
MODEL_FOLDER = os.path.join(APP_ROOT, 'static', 'trained_model')
CNN_MODEL_FILENAME = 'cnn_rnn_model.h5'
RESNET50_MODEL_FILENAME = 'resnet50_model.h5'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

model = keras.models.load_model(os.path.join(MODEL_FOLDER, CNN_MODEL_FILENAME))
resnet_model = ResNet50(weights='imagenet')
resnet_extractor = Model(inputs=resnet_model.input,
						 outputs=resnet_model.get_layer('avg_pool').output)

Time = 0


def start_time():
	global Time
	Time = time.time()


def stop_time():
	return time.time() - Time


def reset_time():
	Time = 0

@app.route('/')
def upload_file():
	return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_files():
	predict = [0] * 5
	class_properties = [""] * 5
	video_filename = ""
	if request.method == 'POST':
		f = request.files['file']
		if os.path.exists(app.config['UPLOAD_FOLDER']) and os.path.isdir(app.config['UPLOAD_FOLDER']):
			shutil.rmtree(app.config['UPLOAD_FOLDER'])
		os.mkdir(app.config['UPLOAD_FOLDER'])
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
		video_filename = "upload/" + f.filename
		images = preprocess_data(f.filename)
		predict = get_prediction(images)
		class_properties[predict.index(max(predict))] = "active-row"
	r = make_response(render_template('result.html', predict=predict, class_properties=class_properties, video_filename=video_filename))
	return r


def preprocess_data(filename):

	start_time()
	images = [[]]
	#generating frames using opencv
	video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
	print("\nPreProcessing time:", stop_time())
	return images


def get_prediction(images):
	start_time()
	resnet_cnn_images = [resnet_extractor.predict(images[0])]
	resnet_cnn_images = np.asarray(resnet_cnn_images)
	input = sequence.pad_sequences(resnet_cnn_images, maxlen=30)
	predict = model.predict(input[0:1])
	s = np.sum(predict)
	print("\n Getting prediction time:", stop_time())
	return [round(p*100/s, 1) for p in predict[0]]

@app.after_request
def add_header(r):

	r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
	r.headers["Pragma"] = "no-cache"
	r.headers["Expires"] = "0"
	# r.headers['Cache-Control'] = 'public, max-age=0'
	return r

if __name__ == '__main__':
   app.run(debug=False)