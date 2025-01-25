import pathlib
import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

#title
st.title('Transportni klassifikatsiya qiluvchi model!')

#rasm joylash
file = st.file_uploader('Rasm yuklash', type= ['png', 'jpeg', 'gif', 'svg'])
st.image(file)

#PIL image
if file is not None:
    img = PILImage.create(file)
else:
    st.warning("Klassifikatsiya uchun avval rasm yuklang!")

#model
model = load_learner('transport_model.pkl')

#prediction
pred, pred_id, probs = model.predict(img)
st.success(f'It is {pred.lower()}!')
st.info(f'Probability: {probs[pred_id]*100:.1f}%')

