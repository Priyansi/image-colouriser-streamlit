import streamlit as st
import matplotlib.pyplot as plt  # To plot the image
import altair as alt  # To plot the label ranking
from PIL import Image
from torch_utils import transform_image, get_prediction

st.title("Image Colouriser")

st.set_option('deprecation.showfileUploaderEncoding', False)

image_data = st.file_uploader("Upload file", type=["jpg", "png", "jpeg"])


def predict(image, m='f'):
    tensor = transform_image(image)
    prediction = get_prediction(tensor, m)
    return prediction


if image_data is not None:
    image = Image.open(image_data)
    st.write("Colourise with model trained on")
    scenary = st.button('Scenary')
    fruits = st.button('Fruits')
    m = 'f' if fruits else 's'
    f, arr = plt.subplots(1, 2, sharey=True)
    arr[0].imshow(transform_image(image).permute(1, 2, 0))
    arr[0].axis('off')
    arr[1].imshow(predict(image, m))
    arr[1].axis('off')
    st.pyplot()
