import streamlit as st
from PIL import Image
from clf import predict_with_resnet101, segment_with_mrcnn
import os
import requests


if os.path.isfile('resnet_scripted.pt')==False:


    URL_RESNET = "https://drive.google.com/file/d/18Rh9mN8FXb-8gsq9KsC9pxE3boqLrpFH/view?usp=sharing"
    response_res = requests.get(URL_RESNET)
    open("resnet_scripted.pt", "wb").write(response_res.content)

if os.path.isfile('mrcnn_scripted.pt')==False:

    URL_MRCNN = "https://drive.google.com/file/d/1E86dc0S3gx8P0Prtm1yqNVG5OuDnOtqX/view?usp=sharing"
    response_mrcnn = requests.get(URL_MRCNN)
    open("mrcnn_scripted.pt", "wb").write(response_mrcnn.content)


app_mode = st.sidebar.selectbox('Select Page',['Model_1', 'Model_2'])

if app_mode=='Model_1':

    file_up = st.file_uploader("Upload an image", type="jpg")

    # image = Image.open(file_up)
    # st.image(image, caption='Uploaded Image.', use_column_width=True)

    if file_up is not None:
        image = Image.open(file_up)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Just a second...")
        labels = predict_with_resnet101(file_up)

        # print out the top 5 prediction labels with scores
        for i in labels:
            st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])



if app_mode=='Model_2':

    file_up = st.file_uploader("Upload an image", type="jpg")

    # image = Image.open(file_up)
    # st.image(image, caption='Uploaded Image.', use_column_width=True)

    if file_up is not None:
        image = Image.open(file_up)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Just a second...")
        img, _, __ = segment_with_mrcnn(file_up)

        st.image(img, caption='Segmented Image.', use_column_width=True)
        
        # print out the top 5 prediction labels with scores
        # for i in labels:
        #     st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
