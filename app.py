import numpy as np
import pyaudio
import struct
import time
from flask import Flask, jsonify, render_template

app = Flask(__name__)

class AudioStream(object):
    def __init__(self):
        # Reduce chunk size to process smaller amounts of data more frequently
        self.CHUNK = 512
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.stream = None
        self.init_audio()
        self.threshold = 0.01
        self.min_rms = 1e-5

    def init_audio(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=False,
            frames_per_buffer=self.CHUNK,
        )

    def get_decibel(self):
        # Read smaller chunks of data for lower latency
        data = self.stream.read(self.CHUNK, exception_on_overflow=False)
        data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)
        data_np = np.array(data_int, dtype='b')[::2] + 128

        rms = np.sqrt(np.mean(np.square(data_np - 128)))

        if rms < self.min_rms:
            rms = self.min_rms

        db = 20 * np.log10(rms)
        return db

# Initialize Audio Stream
audio_stream = AudioStream()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_db_level')
def get_db_level():
    db_level = audio_stream.get_decibel()
    return jsonify({"db_level": db_level})

if __name__ == '__main__':
    # Consider using a multi-threaded server like Gunicorn for production
    app.run(debug=True, threaded=True)
