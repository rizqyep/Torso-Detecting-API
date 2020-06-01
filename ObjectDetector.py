import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

class Detector : 
    def __init__(self):
        #declare model
        global model
        model = keras.models.load_model('torso.h5')

    def detectObject(self, imName):
        x = image.img_to_array(imName) #ubah gambar jadi array 
        x = np.expand_dims(x, axis=0) 
        images = np.vstack([x]) #pecah ke numpy array
        classes = model.predict(images, batch_size=10) # prediksi array dengan model
        result = str(classes[0])
        if result == "[1.]" :
            return "Detected : Torso"
        elif result == "[0.]":
            return "Detected : Manusia"
        

    
    
    
