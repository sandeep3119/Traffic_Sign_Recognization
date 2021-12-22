import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort,send_from_directory
from werkzeug.utils import secure_filename
from flask_dropzone import Dropzone





webapp_root = "webapp"
static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")


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

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(uploaded_file.stream):
            return "Invalid image", 400
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return '', 204

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5050, debug=True)