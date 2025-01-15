import pyaudio
import wave
from pynput import keyboard
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

    def generate_filename(self, prefix, extension):
        """Generate a unique filename using the current date and time."""
        timestamp = datetime.now().strftime("%d-%m-%Y-%H.%M.%S")
        return f"{prefix}-{timestamp}.{extension}"

    def start_recording(self):
        """Start recording audio and stop when Spacebar is pressed."""
        output_file = self.generate_filename("rec", "wav")
        print(f"Recording started. Press 'Spacebar' to stop recording.")
        self.is_recording = True
        self.frames = []
        stream = self.audio.open(format=self.format,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 frames_per_buffer=self.chunk)

        def on_press(key):
            if key == keyboard.Key.space:
                print("\nSpacebar pressed. Stopping the recording...")
                self.is_recording = False
                return False  # Stops the listener

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        while self.is_recording:
            data = stream.read(self.chunk)
            self.frames.append(data)

        stream.stop_stream()
        stream.close()

        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(self.frames))

        print(f"Recording saved as {output_file}")
        return output_file

    def transcribe_audio(self, audio_file):
        """Transcribe the recorded audio using Whisper and save to a text file."""
        print("Transcribing audio...")
        model = load_model("base")
        result = model.transcribe(audio_file, language="en")
        transcription = result["text"]

        output_file = self.generate_filename("transcript", "txt")
        with open(output_file, "w") as f:
            f.write(transcription)

        print(f"Transcription saved as {output_file}")
        # Fix: Added display of transcription here
        print("\nTranscription:\n" + transcription)
        return transcription


def main():
    recorder = VoiceRecorder()

    while True:
        print("\nOptions:")
        print("1. Start Recording")
        print("2. Exit")

        try:
            choice = input("Enter your choice: ").strip()
            if choice == "1":
                audio_file = recorder.start_recording()
                # Ensure transcription prints to terminal
                transcription = recorder.transcribe_audio(audio_file)

            elif choice == "2":
                print("Exiting the program. Goodbye!")
                break
            else:
                print("Invalid input. Please try again.")
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.")


if __name__ == "__main__":
    main()
