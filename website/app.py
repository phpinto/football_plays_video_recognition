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
app = Flask(__name__)

@app.route('/')
def upload_file():
	return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_files():
	print("Tako")
	return render_template('result.html')

		
if __name__ == '__main__':
   app.run(debug = True)