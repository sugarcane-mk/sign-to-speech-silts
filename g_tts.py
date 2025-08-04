from gtts import gTTS
from IPython.display import Audio
from pydub import AudioSegment
import sys


text = sys.argv[1]
print(text)
tts = gTTS(text, lang='ta')
tts.save("output_tamil.mp3")


# Load the MP3 file
audio = AudioSegment.from_mp3("output_tamil.mp3")

# Export as WAV
audio.export("test.wav", format="wav")

# Verify by creating an audio object
#Audio("output_tamil.wav")