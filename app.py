import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer
from src import Detector

st.set_page_config(page_title='Yolo Head Detector', layout='wide')
st.title("Yolo Head Detector using ONNX Runtime")

# Initialize session state only once
if "input" not in st.session_state:
    st.session_state.input = {}

if "start" not in st.session_state:
    st.session_state.start = False

def check_options(option):
    if option is None:
        st.session_state.start = False
        st.warning("Please select an input type first")
        return

    value = st.session_state.input.get(option)

    if not value:
        st.session_state.start = False
        st.warning("No input provided")
        return

    st.session_state.start = True

with st.sidebar:

    input_option = st.radio(label='Select Input', options=['Live', 'Video', 'Image'], index=None)

    if input_option == 'Video':
        video_option = st.radio(label='Select Option', options=['File', 'Path'])
        video = None

        if video_option == 'File':    
            video = st.file_uploader('Upload Video (mp4)', type=['.mp4'])

        else:
            video = st.text_input(label='File Path')

        st.session_state.input[input_option] = video
        
    elif input_option == 'Image':
        image_option = st.radio(label='Select Option', options=['File', 'Path'])
        image = None

        if image_option == 'File':    
            image = st.file_uploader('Upload Video (mp4)', type=['.jpeg', '.jpg', '.png'])

        else:
            image = st.text_input(label='File Path')

        st.session_state.input[input_option] = image

    elif input_option == 'Live':
        st.session_state.input[input_option] = 'cam'

    start_btn = st.button('Start', on_click=lambda: check_options(input_option))

if st.session_state.start:

    for key in st.session_state.input.keys():

        if key == 'Live':
            
            webrtc_streamer(
                key='detectLive',
                video_processor_factory=Detector,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
