import librosa
import numpy as np
import crepe
from collections import Counter

audio_path = "C://Users//adhik//OneDrive//Desktop//indian classical to notation//audio_to_notation//output_folder//bageshree02//vocals.wav"
y, sr = librosa.load(audio_path, sr=16000)

_, freq, confidence, _ = crepe.predict(y, sr, step_size=50, viterbi=True)
pitch_values = freq[confidence > 0.8]
pitch_times = np.linspace(0, len(y) / sr, num=len(freq))
pitch_times = pitch_times[confidence > 0.8]

pitch_log2 = np.log2(pitch_values)
rounded = np.round(pitch_log2 * 12)
mode_pitch_class = Counter(rounded % 12).most_common(1)[0][0]
##auto tonic detection
## detect dynamically
tonic_hz = 146.83* 2 ** ((mode_pitch_class - 9) / 12)
print(f"🎧 Estimated Tonic (Sa): {tonic_hz:.2f} Hz")

sargam_notes = ['Sa', 'Re', 'Re+', 'Ga', 'Ma', 'Ma+', 'Pa', 'Dha', 'Dha+', 'Ni', "Sa'", "Re'"]

def get_sargam_note(freq, tonic):
    if freq <= 0:
        return "-"
    ratio = freq / tonic
    semitones = round(12 * np.log2(ratio))
    index = semitones % 12
    return sargam_notes[index]

sargam_sequence = []
time_sequence = []
for i, f in enumerate(pitch_values):
    note = get_sargam_note(f, tonic_hz)
    sargam_sequence.append(note)
    time_sequence.append(pitch_times[i])

beat_duration = 0.5  ## dynamic 
bar_length = 4       ##dynamic
total_beats = int(time_sequence[-1] / beat_duration)
bars = [[] for _ in range((total_beats // bar_length) + 1)]

for note, t in zip(sargam_sequence, time_sequence):
    beat_index = int(t / beat_duration)
    bar_index = beat_index // bar_length
    bars[bar_index].append(note)

print("\n🪷 Bandish-style Sargam Notation (Teentaal - 4x4):\n")
for i, bar in enumerate(bars):
    bar_display = " | " + "  ".join(bar[:bar_length]) + " " * (4 - len(bar)) + "|"
    if i % 4 == 0:
        print("—" * 40)
    print(bar_display)
print("—" * 40)
