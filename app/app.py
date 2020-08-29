import numpy as np
from torch_utils import transform_image, get_prediction
from PIL import Image
import altair as alt  # To plot the label ranking
import matplotlib.pyplot as plt  # To plot the image
import streamlit as st

st.beta_set_page_config(
    page_icon=":rainbow:",
)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown("# Image Colouriser")

st.set_option('deprecation.showfileUploaderEncoding', False)

image_data = st.file_uploader("Upload file", type=["jpg", "png", "jpeg"])


def predict(tensor, m='f'):
    prediction = get_prediction(tensor, m)
    return prediction


def convert_to_tensor(image):
    tensor = transform_image(image)
    if(tensor.shape[0] == 1):
        tensor = tensor.repeat(3, 1, 1)
    return tensor


if image_data is not None:
    image = Image.open(image_data)
    st.write(
        '<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    model = st.radio("Colourise with model trained on",
                     ('Landscapes', 'Fruits'))
    colourise = st.button('Colourise')
    if colourise:
        m = 'f' if model == 'Fruits' else 's'
        tensor = convert_to_tensor(image)
        f, arr = plt.subplots(1, 2, sharey=True)
        arr[0].imshow(tensor.permute(1, 2, 0))
        arr[0].axis('off')
        arr[0].title.set_text('Before')
        arr[1].imshow(predict(tensor, m))
        arr[1].axis('off')
        arr[1].title.set_text('After')
        st.pyplot()
