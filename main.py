import pickle
import speech_recognition as sr
from first import *

filename = 'model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
recogniser = sr.Recognizer()
with sr.Microphone as source:
    print('Clearing background noise...')
    recogniser.adjust_for_ambient_noise(source, duration=1)
    print('Waitng for your message...')
    recorded_audio = recogniser.listen(source)
    print('Done recording...')

try:
    feature = extract_feature(recorded_audio, mfcc=True, chroma=True, mel=True)
    feature = feature.reshape(1, -1)
    prediction = loaded_model.predict(feature)
    print(prediction)
except Exception as e:
    print(e)

