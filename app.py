import gradio as gr
import cv2
import numpy as np
from src import Detector
import tempfile

detector = Detector()

#--------------IMAGE DETECTION--------------------#
def detect_image(image):
    if not image:
        return
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return detector.detector(image)

#--------------VIDEO DETECTION--------------------#
def detect_video(video_file):
    if not video_file:
        return
    cap = cv2.VideoCapture(video_file)

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    outfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    
    writer = cv2.VideoWriter(outfile.name, fourcc, fps, (w,h), isColor=True)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = detector.detector(frame)
        writer.write(frame[:,:,::-1])
    
    writer.release()
    cap.release
    
    return outfile.name

#--------------LIVE DETECTION--------------------#
def live_detection(image):
    detector.frame_count += 1
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return detector.detector(image)

#---------------GRADIO UI-------------------------#
with gr.Blocks(title='Yolo Head Detector') as app:
    gr.Markdown("# 🎯 YOLO Head Detection (ONNX Runtime)")
    gr.Markdown("### 🔗 Training & Testing Code: [View on GitHub](https://github.com/AbhijithP96/yolo-head-detection)")

    # IMAGE TAB ---------->
    with gr.Tab('Image'):
        with gr.Row():
            input_image = gr.Image(type="numpy", label="Upload Image", sources='upload')
            output_image = gr.Image(label='Detection Result')
        input_image.change(detect_image, input_image, output_image)

    # VIDEO TAB ---------->
    with gr.Tab("Video"):
        with gr.Row():
            input_video = gr.Video(label="Input Video", sources=['upload'], format='webm')
            output_video = gr.Video(label="Detecton Result")
        input_video.change(detect_video, input_video, output_video)
        
    # LIVE TAB ---------->
    with gr.Tab("Live"):
        with gr.Row():
            cam = gr.Image(type='numpy', sources=['webcam'], streaming=True)
            out = gr.Image(label='Live Detection', streaming=True)
            
        cam.stream(live_detection, inputs=[cam], outputs=[out])


    
if __name__ == '__main__':
    app.launch()