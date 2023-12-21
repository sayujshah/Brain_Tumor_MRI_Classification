import os
import cv2
import numpy as np
import gradio as gr
from keras.models import load_model

# Helper function to load a user-chosen pre-trained model
def import_model(model_name):
    # Set aboslute path
    absolute_path = os.path.dirname(__file__)
    model_path = os.path.join(absolute_path, 'Pre-Trained Models/', model_name)
    model = load_model(model_path)

    return model

# Helper function to form model prediction
def img_pred(model, upload):
    mri = cv2.cvtColor(np.array(upload), cv2.COLOR_BGR2RGB)
    mri = cv2.resize(mri, (150, 150))
    mri = mri.reshape(1,150,150,3)
    
    pred = model.predict(mri)
    pred = np.argmax(pred, axis=1)[0]

    return pred

# Use Gradio to develop a GUI to prompt user to upload their own MRI scan
def pred_interface(image):
    model = import_model('effnet_model.keras')
    pred = img_pred(model, image)
    if pred == 0:
        out = 'Glioma Tumor'
    elif pred == 1:
        out = 'Meningioma Tumor'
    elif pred == 2:
        out = 'No tumor detected!'
    else:
        out = 'Pituitary Tumor'

    if pred == 2:
        return out
    else:
        return f'The Model predicts that it is a {out}'

mri_upload = gr.Interface(
    fn=pred_interface,
    inputs=gr.Image(type='pil'),
    outputs='text',
    flagging_options=['Correct', 'Incorrect'],
)

if __name__ == '__main__':
    mri_upload.launch(inbrowser=True)
