
#
#import numpy as np
#from PIL import Image
#from sklearn.preprocessing import LabelEncoder
#from tensorflow.keras.models import load_model
#
#
#def getPrediction(filename):
#
#    classes = ['Negative','Positive']
#    le = LabelEncoder()
#    le.fit(classes)
#    le.inverse_transform([2])
#
#
#    #Load model
#    my_model=load_model("models/mobile_paper_model")
#
#    SIZE = 224 #Resize to same size as training images
#    img_path = 'static/images/'+filename
#    img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
#
#    img = img/255.      #Scale pixel values
#
#    img = np.expand_dims(img, axis=0)  #Get it tready as input to the network
#
#    pred = my_model.predict(img) #Predict
#
#    #Convert prediction to class name
#    pred_class = le.inverse_transform([np.argmax(pred)])[0]
#    print("Diagnosis is:", pred_class)
#    return pred_class
#
#
#
#
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def getPrediction(filename):
    classes = ['Negative', 'Positive']
    
    # Load the pre-trained model
    my_model = load_model("models/mobile_paper_model")
    
    #SIZE = 224  # Resize to the same size as training images
    img_path = 'static/images/' + filename
  # sklearn.image.Load()
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
    
   # img = np.asarray(Image.open(img_path).resize((SIZE, SIZE)))
    
    # Convert grayscale image to 3 channels
    #img = np.stack((img,) * 3, axis=-1)
    
   # img = img / 255.  # Scale pixel values
  #  img = np.expand_dims(img, axis=-1)  # Add a channel dimension
    #img = np.expand_dims(img, axis=0)   # Add a batch dimension
    
    #pred = my_model.predict(img)  # Perform prediction
    
    # Determine the predicted class index
    #predicted_class_index = np.argmax(pred)
    
    
    # Convert class index to class name
    pred_class = classes[result]
    print("Diagnosis is:", pred_class)
    return pred_class

