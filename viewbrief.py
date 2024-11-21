#viewbrief
import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import randomgreedy as rndgreed
import time
 
re_ordered = ''

st.title('Brief Map View')

@st.dialog("Input your File")
def upload():
    uploaded_file = st.file_uploader("Please select your file")
    if st.button("Submit"):
        st.session_state["upload"] = uploaded_file
        st.rerun()
		
if "upload" not in st.session_state:
    st.write("Please Upload your Brief")
    if st.button("Upload File"):
        upload()
else:
    upload_file = st.session_state["upload"]
    print('starting')
    with st.spinner('Please wait...'):
        re_ordered,original, url = rndgreed.mainloop(upload_file)
        time.sleep(1)
    st.header('Re-Ordered Brief:')   
    st.write(original)  
    st.link_button("View Map",url)
    st.write('Or, if you would like to use google MyMaps...')
    st.header('MyMap Layer 1:')   
    st.write(re_ordered[:'J'])
    st.header('MyMap Layer 2:')  
    st.write(re_ordered['J':])

