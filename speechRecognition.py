from deepspeech import Model
import numpy as np
import os
import wave
import subprocess
from IPython.display import Audio

from pytube import YouTube
import os
import torch
from transformers import pipeline

class Transcript:
    def __init__(self,link):
        self.link=link
        
    def youtube_audio(self):
        '''
        
        Extract video from Youtube
        ->Convert the video to audio
        ->Convert audio to 16kHZ format
        
        '''
        if os.path.exists('yt_audio.mp3'):
            os.remove('yt_audio.mp3')
        if os.path.exists('out.wav'):
            os.remove('out.wav')
        yt=YouTube(self.link)
        video = yt.streams.filter(only_audio=True).first()
        output_file = video.download()
        base, ext = os.path.splitext(output_file)
        new_file =  'yt_audio.mp3'
        os.rename(output_file, new_file)

        subprocess.call('ffmpeg -i "yt_audio.mp3" -acodec pcm_s16le -ac 1 -ar 16000 out.wav -y', shell=True)
        
   
    def listen_audio(self):
        '''
        method to listen to the audio file
        '''
        Audio('out.wav')
    
    def run_transcript(self):
        '''
        Method to use the deepSpeech model to 
        generate transcript of the audio file
        '''
        self.youtube_audio()

        model_file_path='deepspeech-0.9.3-models.pbmm'
        lm_file_path='deepspeech-0.9.3-models.scorer'

        beam_width=100
        lm_alpha=0.93
        lm_beta=1.18
        
        model= Model(model_file_path)
        model.enableExternalScorer(lm_file_path)
        
        model.setScorerAlphaBeta(lm_alpha, lm_beta)
        model.setBeamWidth(beam_width)
        buffer, rate= self.read_wav_file()
        data16=np.frombuffer(buffer, dtype=np.int16)
        print(model.stt(data16))
        summarizer = pipeline("summarization",
    "pszemraj/long-t5-tglobal-base-16384-book-summary",
        device=0 if torch.cuda.is_available() else -1,
)
        long_text=model.stt(data16)
        result = summarizer(long_text)
        print("-----summary-----")
        print(result[0]["summary_text"])
        
        
#         return model.stt(data16)
        
    def read_wav_file(self):
        with wave.open('out.wav','rb') as w:
            rate=w.getframerate()
            frames=w.getnframes()
            buffer=w.readframes(frames)
            print("Rate:", rate)
            print("Frames:", frames)
        return buffer, rate   



