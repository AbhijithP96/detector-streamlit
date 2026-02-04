import onnxruntime
import numpy as np
import os
from streamlit_webrtc import VideoProcessorBase
import av
from src.utils import preprocess_image, nms, xywh2xyxy, draw_box

class Detector(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        onnx_model = os.path.join(os.getcwd(), 'weights', 'best.onnx')

        opt_session = onnxruntime.SessionOptions()
        opt_session.enable_mem_pattern = False
        opt_session.enable_cpu_mem_arena = False
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

        EP_list = ['CPUExecutionProvider']
        self.ort_session = onnxruntime.InferenceSession(onnx_model, providers=EP_list)

    def recv(self, frame):
        frame = frame.to_ndarray(format='bgr8')
        #frame = self.detector(frame)

        return av.VideoFrame.from_ndarray(format='bgr8')

    def detector(self, img: np.ndarray):

        ort_session = self.ort_session
        image_height, image_width = img.shape[:2]
        image_draw = img.copy()
        img = preprocess_image(img)

        model_inputs = ort_session.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        input_shape = model_inputs[0].shape
        input_height, input_width = input_shape[2:]

        model_output = ort_session.get_outputs()
        output_names = [model_output[i].name for i in range(len(model_output))]

        outputs = ort_session.run(output_names, {input_names[0] : img})[0]

        predictions = np.squeeze(outputs).T
        conf_thresold = 0.5
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_thresold, :]
        scores = scores[scores > conf_thresold]  

        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = predictions[:, :4]

        input_shape = np.array([input_width, input_height, input_width, input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_width, image_height, image_width, image_height])
        boxes = boxes.astype(np.int32)


        indices = nms(boxes, scores, 0.0)

        for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
            bbox = bbox.round().astype(np.int32).tolist()
            cls_id = int(label)
            cls = 'head' if cls_id == 0 else 'Unknown'
            color = (0,255,0)
            image_draw = draw_box(image_draw, bbox, cls, score, color)
            
        return image_draw
