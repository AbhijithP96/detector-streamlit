from streamlit_webrtc import webrtc_streamer
from src import Detector
import streamlit as st
import cv2
import numpy as np
import tempfile

import threading

# set page config
st.set_page_config(page_title='Yolo Head Detector', page_icon="🎯", layout='wide')
st.title("Yolo Head Detection using ONNX Runtime 🎯", text_alignment='center')

# setting up required objects
lock = threading.Lock()
img_container = {"img": None}

# setup required session states
if "detector" not in st.session_state:
    st.session_state.detector = Detector()
if "input" not in st.session_state:
    st.session_state.input = {}
if "start" not in st.session_state:
    st.session_state.start = False

# helper functions
def callback(frame):
    image = frame.to_ndarray(format='bgr24')

    with lock:
        img_container['img'] = image
        
    return frame

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

def get_uploaded_video(file):
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tmpfile.write(file.read())
    return tmpfile, cv2.VideoCapture(tmpfile.name)

# main ui elements
with st.sidebar:
    input_option = st.radio('Select Input Option', options=['Live', 'Video', 'Image'], index=None)
    if input_option:
        st.session_state.input = {}
        input_option = input_option.lower()

    if input_option == 'live':
        st.session_state.input[input_option] = 'webrtc'

    if input_option == 'video':
        st.session_state.input[input_option] = st.file_uploader(label='Video File', type=['mp4', 'avi', 'mov'])
    
    if input_option == 'image':
        st.session_state.input[input_option] = st.file_uploader(label='Image File', type=['jpg', 'png', 'jpeg', 'bmp'])

    st.button("Start", on_click=lambda: check_options(input_option))

col1, col2 = st.columns([6,6])

if st.session_state.start:

    if 'live' in st.session_state.input.keys():

        with col1:
            ctx = webrtc_streamer(key='streamer',
                            video_frame_callback=callback,
                            media_stream_constraints={"video": True, "audio": False})

        with col2:
            frame_window = st.empty()
            detector = st.session_state.detector

            while ctx.state.playing:

                with lock:
                    img = img_container['img']

                if img is None:
                    continue

                img = detector.detector(img)
                frame_window.image(img)

    if 'video' in st.session_state.input.keys():
        file = st.session_state.input['video']
        if file:
            file, cap = get_uploaded_video(file)

            with col1:
                frame_window_orig = st.empty()
            with col2:
                frame_window_detect = st.empty()
            detector = st.session_state.detector

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_detect = detector.detector(frame)

                frame_window_orig.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_window_detect.image(frame_detect)

            cap.release()

    if 'image' in st.session_state.input.keys():
        file = st.session_state.input['image']
        if file:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            detector = st.session_state.detector

            with col1:
                st.image(img, channels='BGR')

            with col2:
                box = detector.detector(img)
                st.image(box)