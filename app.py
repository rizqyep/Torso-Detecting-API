from ObjectDetector import Detector
import io
import cv2

from flask import Flask, render_template, request

from PIL import Image, ExifTags
from flask import send_file

app = Flask(__name__)

detector = Detector()

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect")
def recog():
    return render_template('recognition.html')


@app.route("/detect", methods=['POST'])
def upload():
    if request.method == 'POST':
        file = Image.open(request.files['file'].stream) #ambil file
        file = file.resize((200,300), Image.ANTIALIAS) #resize , biar sesuai sama target size di tf
        det = Detector() 
        return(det.detectObject(file))#panggil fungsi detect ,parameter file yg di upload atau request via api



