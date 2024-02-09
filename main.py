import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print('Clearing background noise...')
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print('Waiting for message...')
    recorded_audio = recognizer.listen(source)
    print('Done recording...')

try:
    print('Printing the message...')
    text = recognizer.recognize_google_cloud(recorded_audio, language='en-US')
    print(f'Your message: {text}')
except Exception as e:
    print(e)

sentence = [str(text)]
analyser = SentimentIntensityAnalyzer()

for i in sentence:
    v = analyser.polarity_scores(i)