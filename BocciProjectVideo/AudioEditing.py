# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:27:50 2020

@author: rierv
"""

import noisereduce as nr
from scipy.io import wavfile
# load data
from pydub.playback import play
import numpy

#rate, data = wavfile.read("./speech.wav")


#data=data/(2**15)
# select section of data that is noise

# perform noise reduction
#reduced_noise = nr.reduce_noise(audio_clip=data[0], noise_clip=data[0:1][0], verbose=True)
#wavfile.write("edit1.wav", rate, reduced_noise)

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)



from pydub import AudioSegment
from pydub.effects import normalize
from pydub.playback import play
from scipy.signal import wiener
import noisereduce as nr

def adjustVoice (voice):
    # Import target audio file
    song = AudioSegment.from_file(voice)
    #reduced_noise = nr.reduce_noise(audio_clip=song, noise_clip=song, verbose=True)
    
    
    # Normalize target audio file
    #normalized_sound = match_target_amplitude(normalized_loud_then_quiet, 1)
    # boost volume by 6dB
    normalized_loud_then_quiet = normalize(song, -5)
    
    #lastsong=wiener(louder_song, 1)
    # reduce volume by 3dB
    quieter_song = song - 3
    
    out=normalized_loud_then_quiet.low_pass_filter(20) +20
    
    #Play song
    play(out)
    out.export(voice, format="wav")