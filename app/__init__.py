from flask import Flask, request, jsonify,  redirect, url_for, send_from_directory
from PIL import Image
import secrets
from os.path import join as pjoin
from app import ia

app = Flask(__name__)


@app.route('/')
def index():
    return "<h1>passou</h1>"

@app.route("/im_size", methods=["POST"])
def process_image():
    
    
    file = request.files['image']
    # Read the image via file.stream

    img = Image.open(file.stream)
    pokemon = ia.modelpredict(img)
    return jsonify({"pokemon":pokemon})



