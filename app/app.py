import streamlit as st
import matplotlib.pyplot as plt  # To plot the image
import altair as alt  # To plot the label ranking
import io
from PIL import Image
from app.torch_utils import transform_image, get_prediction

st.title("Image Colouriser")

st.set_option('deprecation.showfileUploaderEncoding', False)

image_data = st.file_uploader("Upload file", type=["jpg", "png", "jpeg"])


def import_and_predict(image_data):
    if image_data is not None:
        image = Image.open(image_data)
        tensor = transform_image(image)
        prediction = get_prediction(tensor)
        return prediction


if image_data is None:
    st.text("Please upload an image file")
else:
    prediction = import_and_predict(image_data)
    st.image(prediction, use_column_width=True)
