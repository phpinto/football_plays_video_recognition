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
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'upload')
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
	return render_template('result.html')

if __name__ == '__main__':
   app.run(debug = True)