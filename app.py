import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import subprocess
import nemo.collections.asr as nemo_asr
import openai

# Helper: Convert audio to wav if needed
def convert_to_wav(input_path):
    if input_path.lower().endswith('.wav') or input_path.lower().endswith('.flac'):
        return input_path
    output_path = input_path + '.converted.wav'
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ar', '16000', '-ac', '1', output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path

# Helper: Transcribe audio using Parakeet
def transcribe_audio(audio_path):
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    output = asr_model.transcribe([audio_path])
    return output[0].text

# Helper: Generate notes using OpenAI
def generate_notes(transcription, api_key):
    openai.api_key = api_key
    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Create notes from this transcription."},
            {"role": "user", "content": transcription}
        ]
    )
    return response.choices[0].message.content.strip()

# Main App
class AudioToNotesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio to Notes")
        self.audio_path = None
        self.api_key = os.getenv("OPENAI_API_KEY", "")

        tk.Label(root, text="Title:").grid(row=0, column=0, sticky="e")
        self.title_entry = tk.Entry(root, width=40)
        self.title_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Button(root, text="Select Audio File", command=self.select_file).grid(row=1, column=0, columnspan=2, pady=5)
        self.file_label = tk.Label(root, text="No file selected.")
        self.file_label.grid(row=2, column=0, columnspan=2)

        tk.Label(root, text="OpenAI API Key:").grid(row=3, column=0, sticky="e")
        self.api_entry = tk.Entry(root, width=40, show="*")
        self.api_entry.insert(0, self.api_key)
        self.api_entry.grid(row=3, column=1, padx=5, pady=5)

        tk.Button(root, text="Transcribe & Generate Notes", command=self.process).grid(row=4, column=0, columnspan=2, pady=10)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Audio Files", "*.wav *.flac *.mp4a *.mp3 *.ogg *.m4a *.aac *.wma *.opus"),
            ("All Files", "*.*")
        ])
        if path:
            self.audio_path = path
            self.file_label.config(text=os.path.basename(path))

    def process(self):
        title = self.title_entry.get().strip()
        if not title:
            messagebox.showerror("Error", "Please enter a title.")
            return
        if not self.audio_path:
            messagebox.showerror("Error", "Please select an audio file.")
            return
        api_key = self.api_entry.get().strip()
        if not api_key:
            messagebox.showerror("Error", "Please enter your OpenAI API key.")
            return
        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{title}_{dt}"
        try:
            messagebox.showinfo(
                "Transcribing",
                "Transcription in progress. The progress bar in the terminal may not update, but the app is still working. Please wait until the process completes."
            )
            wav_path = convert_to_wav(self.audio_path)
            transcription = transcribe_audio(wav_path)
            trans_file = f"{base}-transcription.txt"
            with open(trans_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            notes = generate_notes(transcription, api_key)
            notes_file = f"{base}-notes.txt"
            with open(notes_file, "w", encoding="utf-8") as f:
                f.write(notes)
            messagebox.showinfo("Success", f"Transcription and notes saved as:\n{trans_file}\n{notes_file}")
            self.root.quit()  # Close the GUI after completion
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioToNotesApp(root)
    root.mainloop()
