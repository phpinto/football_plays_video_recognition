# from flask import Flask, render_template, request, make_response, jsonify


# app = Flask(__name__, static_folder='static')

# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/upload-label',methods=[ "GET",'POST'])
# def uploadLabel():
#     return render_template('result.html')



# if __name__ == '__main__':
#     # app.run(port=5002, debug=True)

#     # Serve the app with gevent
# 	app.run()

from flask import Flask, render_template, request
import os
import shutil
import cv2
import math
from PIL import Image as img
import numpy as np
import matplotlib.pyplot as plt

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'upload')
INPUT_FRAME_FOLDER = os.path.join(UPLOAD_FOLDER, 'input_frames')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def upload_file():
	return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_files():
	if request.method == 'POST':
		f = request.files['file']
		if os.path.exists(app.config['UPLOAD_FOLDER']) and os.path.isdir(app.config['UPLOAD_FOLDER']):
			shutil.rmtree(app.config['UPLOAD_FOLDER'])
		os.mkdir(app.config['UPLOAD_FOLDER'])
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], "input.mp4"))
		preprocess_data()
	return render_template('result.html')


def preprocess_data():
	images = [[]]
	#generating frames using opencv
	video_path = os.path.join(app.config['UPLOAD_FOLDER'], "input.mp4")
	if os.path.exists(video_path):
		cap = cv2.VideoCapture(video_path)
		frameRate = cap.get(cv2.CAP_PROP_FPS)
		x = 1
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
		print(images[0].shape)
		for i in range(len(images[0])):
			plt.figure()
			plt.imshow(images[0][i])
			plt.show()
	return images


if __name__ == '__main__':
   app.run(debug = True)