import numpy as np
from torch_utils import transform_image, get_prediction
from PIL import Image
import altair as alt  # To plot the label ranking
import matplotlib.pyplot as plt  # To plot the image
import streamlit as st
import requests
from io import BytesIO
import base64

st.beta_set_page_config(
    page_title="Image Colouriser",
    page_icon=":rainbow:",
)


hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown("# Image Colouriser")

st.set_option('deprecation.showfileUploaderEncoding', False)


def predict(tensor, m='f'):
    prediction = get_prediction(tensor, m)
    return prediction


def convert_to_tensor(image):
    tensor = transform_image(image)
    if(tensor.shape[0] == 1):
        tensor = tensor.repeat(3, 1, 1)
    return tensor


def get_image_download_link(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a download="result.png" href="data:file/jpg;base64,{img_str}">Download result</a>'
    return href


def plot_results(tensor, prediction):
    f, arr = plt.subplots(1, 2, sharey=True)
    arr[0].imshow(tensor.permute(1, 2, 0))
    arr[0].axis('off')
    arr[0].title.set_text('Before')
    arr[1].imshow(prediction)
    arr[1].axis('off')
    arr[1].title.set_text('After')
    st.pyplot()


image_data = st.file_uploader("Upload file", type=["jpg", "png", "jpeg"])

st.markdown("<p style='text-align: center;'>OR</p>",
            unsafe_allow_html=True)

image_url = st.text_input("URL : ")


if image_data is None and image_url:
    try:
        response = requests.get(image_url)
        image_data = BytesIO(response.content)
    except:
        st.write("Please enter a valid URL")

if image_data is not None:
    try:
        image = Image.open(image_data)
        st.write(
            '<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        model = st.radio("Colourise with model trained on",
                         ('Landscapes', 'Fruits'))
        colourise = st.button('Colourise')
        if colourise:
            m = 'f' if model == 'Fruits' else 's'
            tensor = convert_to_tensor(image)
            prediction = predict(tensor, m)
            result = Image.fromarray((prediction*255).astype(np.uint8))
            # download = st.button('Download Result')
            # if download:
            st.markdown(get_image_download_link(
                result), unsafe_allow_html=True)
            plot_results(tensor, prediction)
    except:
        st.write("Error opening image. Please try again.")
