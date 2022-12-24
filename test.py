import streamlit as st
from PIL import Image
from speechRecognition import Transcript



image= Image.open('yt_logo.png')

st.title('Summarize audio file')
st.image(image)
st.write('Hello Top G')
link=st.text_input(label='Paste youtube link')

if st.button('Get Transcript'):
    summary = Transcript(link).run_transcript()
    st.success(summary)
