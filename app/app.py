import numpy as np
from torch_utils import transform_image, transform_tensor_pil, get_prediction
from PIL import Image
import matplotlib.pyplot as plt  # To plot the before and after image
import streamlit as st
import requests  # For extracting image from URLs
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


def get_image_download_link(img, img_type):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a download="{img_type}.png" href="data:file/jpg;base64,{img_str}">Download {img_type}</a>'
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
        # model = st.sidebar.selectbox("Choose Model :",
        #                              ('Landscapes', 'Animals', 'Fruits', 'People'))
        model = st.sidebar.selectbox("Choose Model :",
                                     ('Landscapes', 'Fruits', 'People'))
        colourise = st.sidebar.button('Colourise')
        if colourise:
            # if model == 'Animals':
            #     m = 'a'
            if model == 'Fruits':
                m = 'f'
            elif model == 'People':
                m = 'p'
            else:
                m = 'l'
            tensor = convert_to_tensor(image)
            prediction = predict(tensor, m)
            result = Image.fromarray((prediction*255).astype(np.uint8))
            st.sidebar.markdown(get_image_download_link(
                transform_tensor_pil(tensor), 'original'), unsafe_allow_html=True)
            st.sidebar.markdown(get_image_download_link(
                result, 'result'), unsafe_allow_html=True)
            plot_results(tensor, prediction)
    except:
        st.write("Error opening image. Please try again.")
