import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort,send_from_directory,jsonify
from flask.helpers import flash
from flask.templating import render_template_string
from werkzeug.utils import secure_filename
from flask_dropzone import Dropzone
from prediction_service.prediction import predict_sign




webapp_root = "webapp"
static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")
prediction_file= os.path.join('prediction_service','prediction.txt')

app = Flask(__name__,static_folder=static_dir,template_folder=template_dir)
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png','jpeg']
app.config['UPLOAD_PATH'] = 'webapp/static/uploads'


dropzone = Dropzone(app)
def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files)
@app.route('/prediction',methods=['GET'])
def indexPrediction():
    file1 = open(prediction_file, "r")
    content=file1.read()
    file1.close()
    return render_template('index.html',pred=content)

@app.route('/predict', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(uploaded_file.stream):
            return "Invalid image", 400
        image_path=uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], 'image'))
        prediction=predict_sign(os.path.join(app.config['UPLOAD_PATH'], 'image'))
        file1 = open(prediction_file, "w")  # write mode
        file1.write(prediction)
        file1.close()
    return prediction,200

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

if __name__ == "__main__":
    app.secret_key='1234'
    app.config['SESSION_TYPE'] = 'prediction'
    app.run(host='127.0.0.1', port=5050, debug=True)