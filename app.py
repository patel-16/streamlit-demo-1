import streamlit as st
from PIL import Image
from clf import predict_with_resnet101

app_mode = st.sidebar.selectbox('Select Page',['Model_1'])

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