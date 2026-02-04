import gradio as gr

def do_something(image):
    print(image.shape)
    yield image

with gr.Blocks(title='Testing Live') as test:
    
    cam = gr.Image(type='numpy', sources=['webcam'], streaming=True)
    out = gr.Image(label='Something', streaming=True)
    
    cam.stream(do_something, inputs=[cam], outputs=[out])    
    
test.launch()