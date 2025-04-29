import librosa
import numpy as np
import crepe
from collections import Counter
from scipy.ndimage import median_filter
from scipy import signal  # Add this import for low-pass filtering

# Load audio
audio_path = "C://Users//adhik//OneDrive//Desktop//indian classical to notation//audio_to_notation//output_folder//bageshree02//Baadal Ghumad Aaye- Sargam notes.wav"
y, sr = librosa.load(audio_path, sr=16000)

# Pitch detection
_, freq, confidence, _ = crepe.predict(y, sr, step_size=50, viterbi=True)
pitch_values = freq[confidence > 0.8]
pitch_times = np.linspace(0, len(y) / sr, num=len(freq))
pitch_times = pitch_times[confidence > 0.8]

# Smooth pitch values
pitch_values = median_filter(pitch_values, size=3)

# Low-pass filter to isolate the drone (tanpura typically 100–300 Hz)
cutoff = 300  # Cutoff frequency in Hz
order = 5  # Filter order
nyquist = 0.5 * sr
normal_cutoff = cutoff / nyquist
b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
y_low = signal.lfilter(b, a, y)  # Apply low-pass filter

# Pitch detection on low-pass filtered audio
_, freq_low, confidence_low, _ = crepe.predict(y_low, sr, step_size=50, viterbi=True)
freq_low = freq_low[confidence_low > 0.8]

# Find the most common frequency (likely the drone, i.e., Sa)
freq_counts = Counter(np.round(freq_low, 1)).most_common(1)
if freq_counts:
    tonic_hz = freq_counts[0][0]
else:
    # Fallback to original method
    pitch_log2 = np.log2(pitch_values)
    rounded = np.round(pitch_log2 * 12)
    mode_pitch_class = Counter(rounded % 12).most_common(1)[0][0]
    tonic_hz = 440.0 * 2 ** ((mode_pitch_class - 9) / 12)
print(f"🎧 Estimated Tonic (Sa): {tonic_hz:.2f} Hz")

# Rest of the code remains the same
sargam_notes = ['Sa', 'Re', 'Ga', 'Ma', 'Pa', 'Dha', 'Ni', "Sa'"]
notation_ids = ['S', 'r', 'g', 'm', 'P', 'd', 'N', "S'"]

def get_sargam_note(freq, tonic):
    if freq <= 0:
        return "-", "-"
    ratio = freq / tonic
    semitones = 12 * np.log2(ratio)
    semitone_positions = [0, 1, 2, 4, 7, 8, 10, 12]  # Sa, komal Re, komal Ga, Ma, Pa, komal Dha, Ni, Sa'
    closest_semitone = min(range(len(semitone_positions)), key=lambda i: abs(semitone_positions[i] - semitones))
    octave = int(semitones // 12)
    if octave >= 1 and closest_semitone == 0:
        return "Sa'", "S'"
    elif octave >= 1:
        return sargam_notes[closest_semitone] + "'", notation_ids[closest_semitone] + "'"
    return sargam_notes[closest_semitone], notation_ids[closest_semitone]

# Generate sequences
sargam_sequence = []
notation_id_sequence = []
time_sequence = []
for i, f in enumerate(pitch_values):
    sargam_note, notation_id = get_sargam_note(f, tonic_hz)
    sargam_sequence.append(sargam_note)
    notation_id_sequence.append(notation_id)
    time_sequence.append(pitch_times[i])

# Organize into bars
beat_duration = 0.5
bar_length = 4
total_beats = int(time_sequence[-1] / beat_duration)
bars_sargam = [[] for _ in range((total_beats // bar_length) + 1)]
bars_notation = [[] for _ in range((total_beats // bar_length) + 1)]

for sargam, notation, t in zip(sargam_sequence, notation_id_sequence, time_sequence):
    beat_index = int(t / beat_duration)
    bar_index = beat_index // bar_length
    bars_sargam[bar_index].append(sargam)
    bars_notation[bar_index].append(notation)

# Display both sargam and notation ID formats
print("\n🪷 Bandish-style Sargam Notation (Teentaal - 4x4):\n")
for i, bar in enumerate(bars_sargam):
    bar_display = " | " + "  ".join(bar[:bar_length]) + " " * (4 - len(bar)) + "|"
    if i % 4 == 0:
        print("—" * 40)
    print(bar_display)
print("—" * 40)

print("\n🪷 Bandish-style Notation IDs (Teentaal - 4x4):\n")
for i, bar in enumerate(bars_notation):
    bar_display = " | " + "  ".join(bar[:bar_length]) + " " * (4 - len(bar)) + "|"
    if i % 4 == 0:
        print("—" * 40)
    print(bar_display)
print("—" * 40)