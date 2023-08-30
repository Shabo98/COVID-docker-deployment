
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def getPrediction(filename):
    classes = ['Negative', 'Positive']
    
    # Load the pre-trained model
    my_model = load_model("models/mobile_paper_model")

    img_path = 'static/images/' + filename

    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    
    print(img_array.shape)
    img_array= np.stack((img_array,) * 3, axis=-1)
    print(img_array.shape)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array.shape
    pred = my_model.predict(img_array)
    
    
    if pred >= 0.50 :
     result = 1
    elif pred < 0.50 :
     result = 0
    
    
    # Convert class index to class name
    pred_class = classes[result]
    print("Diagnosis is:", pred_class)
    return pred_class

