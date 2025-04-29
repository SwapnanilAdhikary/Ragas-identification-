import librosa
import numpy as np
import crepe
from collections import Counter

audio_path = "C://Users//adhik//OneDrive//Desktop//indian classical to notation//audio_to_notation//output_folder//bageshree02//accompaniment.wav"
y, sr = librosa.load(audio_path, sr=16000)  # crepe prefers 16kHz
_, freq, confidence, _ = crepe.predict(y, sr, step_size=50, viterbi=True)
pitch_values = freq[confidence > 0.8]
pitch_times = np.linspace(0, len(y) / sr, num=len(freq))
pitch_times = pitch_times[confidence > 0.8]

# --- Estimate tonic (Sa) ---
pitch_log2 = np.log2(pitch_values)
rounded = np.round(pitch_log2 * 12)
mode_pitch_class = Counter(rounded % 12).most_common(1)[0][0]
tonic_hz = 440.0 * 2 ** ((mode_pitch_class - 9) / 12)
print(f"Estimated tonic: {tonic_hz:.2f} Hz")

# --- Define Sargam scale ---
sargam_notes = ['Sa', 'Re', 'Re+', 'Ga', 'Ma', 'Ma+', 'Pa', 'Dha', 'Dha+', 'Ni', 'Sa\'', 'Re\'']

def get_sargam_note(freq, tonic):
    if freq <= 0:
        return None
    ratio = freq / tonic
    semitones = round(12 * np.log2(ratio))
    index = semitones % 12
    octave = semitones // 12
    return f"{sargam_notes[index]}"

# --- Generate Sargam notation ---
sargam_sequence = []
for f in pitch_values:
    note = get_sargam_note(f, tonic_hz)
    sargam_sequence.append(note)

# --- Group consecutive duplicates ---
transcription = []
prev = None
for note in sargam_sequence:
    if note != prev:
        transcription.append(note)
        prev = note

print("\n🪷 Sargam Notation (Simplified):")
print(" ".join(transcription))

