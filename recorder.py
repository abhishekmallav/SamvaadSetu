# recorder.py

import pyaudio
import wave
from whisper import load_model
from datetime import datetime

class VoiceRecorder:
    def __init__(self):
        self.is_recording = False
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def generate_filename(self, prefix, extension):
        timestamp = datetime.now().strftime("%d-%m-%Y-%H.%M.%S")
        return f"{prefix}-{timestamp}.{extension}"

    def start_recording(self):
        output_file = self.generate_filename("rec", "wav")
        print(f"Recording started. Press 'Stop Recording' to stop.")
        self.is_recording = True
        self.frames = []
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk)

        while self.is_recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

        return output_file

    def stop_recording(self):
        if not self.is_recording:
            return None
        
        print("Stopping the recording...")
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()

    def save_audio(self, output_file):
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(self.frames))

    def transcribe_audio(self, audio_file):
        print("Transcribing audio...")
        model = load_model("base")
        result = model.transcribe(audio_file, language="en")
        transcription = result["text"]

        output_file = self.generate_filename("transcript", "txt")
        with open(output_file, "w") as f:
            f.write(transcription)

        print(f"Transcription saved as {output_file}")
        print("\nTranscription:\n" + transcription)
        return transcription
