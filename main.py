import pickle
import speech_recognition as sr
from model import *

filename = 'model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
recogniser = sr.Recognizer()
with sr.Microphone() as source:
    print('Clearing background noise...')
    recogniser.adjust_for_ambient_noise(source, duration=1)
    print('Waitng for your message...')
    recorded_audio = recogniser.listen(source)
    print('Done recording...')

with open('result.wav', 'wb') as f:
    f.write(recorded_audio.get_wav_data())

feature = extract_feature("result.wav", mfcc=True, chroma=True, mel=True)
feature = feature.reshape(1, -1)
prediction = loaded_model.predict(feature)
print(f"Prediction: {prediction}")

