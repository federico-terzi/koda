import os
import cv2
import koda
import glob
import json
import hashlib
import numpy as np
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, abort, send_file, send_from_directory, render_template

UPLOAD_DIR = "/tmp/upload"
RESULT_DIR = "/tmp/result"

TEMPLATE_DIR = "./web/templates"
print(TEMPLATE_DIR)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create media folder if not present
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

app = Flask(__name__, template_folder = TEMPLATE_DIR)

def list_images():
    images = []
    for file in glob.glob(os.path.join(UPLOAD_DIR, "*")):
        name = os.path.basename(file)
        images.append({
            "id": name, 
            "src": f"/api/get_image?id={name}&type=thumb",
        })
    return images

@app.route('/')
def index():
    images = list_images()
    return render_template("index.html", images = images)

@app.route('/image')
def image():
    iid = request.args.get('id')
    return render_template("image.html", iid = iid)

@app.route('/<path:path>')
def files(path):
    return send_from_directory('web', path)

@app.route('/api/count')
def count():
    images = list_images()
    return str(len(images))

@app.route('/api/get_image', methods=["GET"])
def get_image():
    iid = request.args.get('id')
    itype = request.args.get('type')
    target = os.path.join(RESULT_DIR, f"{iid}_{itype}.png")
    print(target)
    if os.path.exists(target):
        return send_file(target)
    
    abort(404)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/process', methods=['POST'])
def process():
    # check if the post request has the file part
    if 'file' not in request.files:
        abort(500)
        return "O"
    file = request.files['file']
    word = request.form["word"]
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        abort(500)
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = hashlib.md5(file.filename.encode('utf-8')).hexdigest()
        file_path = os.path.join(UPLOAD_DIR, filename)
        file.save(file_path)
        print(file_path)
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        
        doc, imgs = koda.load(img)

        thumbnail = cv2.resize(img, (256, 256))

        cv2.imwrite(os.path.join(RESULT_DIR, f"{filename}_original.png"), img)
        cv2.imwrite(os.path.join(RESULT_DIR, f"{filename}_thumb.png"), thumbnail)
        cv2.imwrite(os.path.join(RESULT_DIR, f"{filename}_edges.png"), imgs['edges'])
        cv2.imwrite(os.path.join(RESULT_DIR, f"{filename}_hough_lines.png"), imgs['hough_lines'])
        cv2.imwrite(os.path.join(RESULT_DIR, f"{filename}_corners.png"), imgs['corners'])
        cv2.imwrite(os.path.join(RESULT_DIR, f"{filename}_warp.png"), imgs['warp'])

        highlight = doc.findWord(word)

        cv2.imwrite(os.path.join(RESULT_DIR, f"{filename}_highlight.png"), highlight)

        return redirect("/")

"""
# Read image

"""