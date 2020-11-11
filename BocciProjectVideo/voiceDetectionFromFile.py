# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:57:35 2020

@author: rierv
"""

import speech_recognition as sr
import text2emotion as te
from googletrans import Translator


def textFromAudio(audio):
    recognizer = sr.Recognizer()
    
    ''' recording the sound '''
    
    with sr.AudioFile(audio) as source:
        recorded_audio = recognizer.listen(source)
    text=""
    ''' Recorgnizing the Audio '''
    try:
        print("Recognizing the text")
        text = recognizer.recognize_google(
                recorded_audio, 
                language="it-IT"
            )
        print("Decoded text : {}".format(text))
        getEmotion(text)

    except Exception as ex:
        print(ex)
    return text

def getEmotion(text):
    translator = Translator()
    
    result = translator.translate(text, dest='en')

    print("Tranlsated text:",result.text)

    print("Text emotion:",te.get_emotion(result.text))
