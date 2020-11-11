# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:01:31 2020

@author: rierv
"""

import pyaudio
import os
import wave
import pickle
from sys import byteorder
from array import array
from struct import pack
from sklearn.neural_network import MLPClassifier

import voiceDetectionFromFile as vff
import AudioEditing as ae
import EmotionFromVoiceModelBuilder as evm
#from python_utils import extract_feature

#import numpy as np
import cv2

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 12384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.
    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)
    
    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        #video
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            id,conf=rec.predict(gray[y:y+h,x:x+w])
            if(id==0):
                id="neutral"
            if id==1:
                id="happy"
            if id==2:
                id="sad"
            if id==3:
                id="angry"
            print(id)
        #endvideo
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break
        
    
    
    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()



if __name__ == "__main__":
    #video
    face_cascade = cv2.CascadeClassifier("C:/Users/rierv/anaconda3/Library/etc/haarcascades/haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    rec = cv2.face.LBPHFaceRecognizer_create() 
    rec.read("C:/Users/rierv/Desktop/Python/trainingdata.yml")
    id=0
    #cv2.destroyAllWindows()
    #endvideo
    # load the saved model (after training)
    model = pickle.load(open("result/mlp_classifier.model", "rb"))
    print("Please talk")
    filename = "testVoiceEmotionRecognizer.wav"
    # record the file (start talking)
    record_to_file(filename)
    cap.release()
    # extract features and reshape it
    features = evm.extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
    print("Voice emotion: { 'Calm'':", round(model.predict_proba(features)[0][1],2),
    "'Happy':", round(model.predict_proba(features)[0][2],2),
    "'Angry':",round(model.predict_proba(features)[0][0],2),
    "'Surprised':", round(model.predict_proba(features)[0][4],2),
    "'Sad':", round(model.predict_proba(features)[0][3],2),"}")
    result = model.predict(features)[0]
    # show the result !
    print("Main voice emotion:", result)
    ae.adjustVoice(filename)
    vff.textFromAudio(filename)