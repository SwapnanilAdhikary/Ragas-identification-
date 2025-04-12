import librosa
import numpy as np

SARGAM_MAP = {
    'Sa': 1.0,
    'Re_komal': 1.066,
    'Re': 1.122,
    'Ga_komal': 1.189,
    'Ga': 1.26,
    'Ma': 1.335,
    'Ma_tivra': 1.414,
    'Pa': 1.5,
    'Dha_komal': 1.587,
    'Dha': 1.682,
    'Ni_komal': 1.782,
    'Ni': 1.888,
    'Sa\'': 2.0  # Higher octave Sa
}

def get_closest_note(pitch, sa_freq):
    if np.isnan(pitch):
        return None
    ratio = pitch / sa_freq
    closest_note = min(SARGAM_MAP.items(), key=lambda x: abs(x[1] - ratio))
    return closest_note[0]

def extract_sargam(file_path):
    print(f"Loading: {file_path}")
    y, sr = librosa.load(file_path, sr=None)

    print("Extracting pitch...")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

    voiced_pitches = f0[voiced_flag]
    sa = np.median(voiced_pitches)
    print(f"Estimated Sa (tonic) frequency: {sa:.2f} Hz")

    times = librosa.times_like(f0)
    note_sequence = []

    for t, pitch in zip(times, f0):
        note = get_closest_note(pitch, sa)
        if note:
            note_sequence.append((round(t, 2), note))

    return note_sequence

if __name__ == "__main__":
    audio_file = "sarang16.wav"  
    notes = extract_sargam(audio_file)
    
    print("\nSargam Timeline:")
    for t, note in notes:
        print(f"{t:.2f}s - {note}")
