# app.py

from flask import Flask, render_template, jsonify
from recorder import VoiceRecorder

app = Flask(__name__)
recorder = VoiceRecorder()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    recorder.start_recording()
    return jsonify(status="Recording started")

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    audio_file = recorder.generate_filename("rec", "wav")  # Temporary filename for saving audio
    recorder.stop_recording()
    recorder.save_audio(audio_file)  # Save recorded audio to file
    transcription = recorder.transcribe_audio(audio_file)
    
    return jsonify(transcription=transcription, audio_file=audio_file)

if __name__ == "__main__":
    app.run(debug=True)
