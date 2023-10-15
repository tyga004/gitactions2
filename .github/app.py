import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2

def main():
    # set up the Streamlit app
    st.write("Name: Tyrone Brix R. Garraton")
    st.write("Section: CPE32S2")
    st.write("Instructor: Engr. Roman Richard")
    st.title("Classifies Flying Vehicle ( Airplane/Helicopter)")
    st.write("This app classifies whether an uploaded image contains a Airplane or Helicopter images.")
   
    @st.cache_resource
    def load_model():
        model = tf.keras.models.load_model('weights-improvement-10-0.98.hdf5')
        return model
    
    def import_and_predict(image_data, model):
        size=(128,128)
        image = ImageOps.fit(image_data,size, Image.LANCZOS)
        image = np.asarray(image)
        image = image / 255.0
        img_reshape = np.reshape(image, (1, 128, 128, 3))
        prediction = model.predict(img_reshape)
        return prediction

    model = load_model()
    class_names = ["AIRPLANE", "HELICOPTER"]
    

    file = st.file_uploader("Choose a Airplane or Helicopter picture from your computer", type=["jpg", "png", "jpeg"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        string = "Prediction: " + class_name
        st.success(string)
 
if __name__ == "__main__":
    main()
