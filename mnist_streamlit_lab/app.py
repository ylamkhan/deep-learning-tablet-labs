"""Tablet-friendly MNIST handwriting recognition web interface."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from preprocessing import prepare_digit

st.set_page_config(page_title='Digit Recognition Lab', page_icon='✍️', layout='centered')
st.markdown('''<style>
.block-container {max-width: 820px; padding-top: 1.5rem;}
.hero {background:linear-gradient(120deg,#17253e,#243f69);padding:22px 26px;border-radius:18px;color:white;margin-bottom:20px}
.hero h1 {font-size:2rem; margin:0 0 8px; color:white}
.hero p {margin:0;color:#d9e7ff}
.result {background:#edf5ff;color:#162b4c;border-radius:16px;padding:20px;text-align:center}
.result .digit {font-size:5rem;font-weight:800;line-height:1.1}
.stButton>button {border-radius:11px;min-height:44px}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>✍️ Handwritten Digit Recognition</h1><p>Deep Learning Lab · CNN + MNIST + Streamlit</p></div>', unsafe_allow_html=True)
st.write('Draw **one digit from 0 to 9** using your finger, stylus or mouse. Then click **Recognize**.')

MODEL_PATH = Path(__file__).parent / 'model' / 'mnist_cnn.keras'
@st.cache_resource
def load_cnn(path):
    return tf.keras.models.load_model(path)

if not MODEL_PATH.exists():
    st.error('Model not found. Run `python train.py` first, or copy your trained model to `model/mnist_cnn.keras`.')
    st.stop()

model = load_cnn(str(MODEL_PATH))
pen_size = st.slider('Pen thickness', 12, 32, 20, 2)
st.caption('Write a large, clear digit in the center of the black box.')
canvas = st_canvas(
    fill_color='rgba(255,255,255,1)', stroke_width=pen_size,
    stroke_color='#FFFFFF', background_color='#000000',
    update_streamlit=True, height=280, width=320,
    drawing_mode='freedraw', return_image_data=True,
    key='digit_canvas',
)
st.caption('To start over, use the canvas Delete/Clear control. On a tablet, use the browser in landscape if necessary.')
if st.button('🔍 Recognize', type='primary', use_container_width=True):
    if canvas.image_data is None:
        st.warning('Draw a digit before predicting.')
    else:
        tensor, processed = prepare_digit(canvas.image_data)
        if tensor is None:
            st.warning('Canvas is empty. Draw a digit first.')
        else:
            probabilities = model.predict(tensor, verbose=0)[0]
            digit = int(np.argmax(probabilities))
            confidence = float(probabilities[digit] * 100)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f'<div class="result"><div>Predicted digit</div><div class="digit">{digit}</div><div>Confidence: {confidence:.1f}%</div></div>', unsafe_allow_html=True)
            with col2:
                st.image(processed.resize((168,168)), caption='CNN input (28×28)', width=168)
            st.subheader('Probability for each class')
            df = pd.DataFrame({'Digit': [str(i) for i in range(10)], 'Probability': probabilities})
            st.bar_chart(df.set_index('Digit'))
            if confidence < 70:
                st.info('Uncertain prediction: try writing a clearer digit.')
            st.caption('The confidence is a softmax score, not a guarantee of correctness.')

with st.expander('How does the app work?'):
    st.markdown('**Canvas → crop and resize → center digit → normalize → CNN → softmax probabilities.**')
    st.markdown('The model sees an image with shape **(28, 28, 1)**. Drawings that differ from MNIST can be misclassified.')
