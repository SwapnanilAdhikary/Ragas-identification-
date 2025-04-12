import librosa
import numpy as np
from collections import Counter
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
    'Sa\'': 2.0
}

RAGA_RULES = {
    "Yaman": {"notes": {'Ni', 'Re', 'Ga', 'Ma_tivra', 'Dha', 'Sa', 'Pa'}},
    "Bhairav": {"notes": {'Sa', 'Re_komal', 'Ga', 'Ma', 'Pa', 'Dha_komal', 'Ni'}},
    "Bhairavi": {"notes": {'Sa', 'Re_komal', 'Ga_komal', 'Ma', 'Pa', 'Dha_komal', 'Ni_komal'}},
    "Kafi": {"notes": {'Sa', 'Re', 'Ga_komal', 'Ma', 'Pa', 'Dha', 'Ni_komal'}},
}

def get_closest_note(pitch, sa_freq):
    if np.isnan(pitch):
        return None
    ratio = pitch / sa_freq
    closest_note = min(SARGAM_MAP.items(), key=lambda x: abs(x[1] - ratio))
    return closest_note[0]

def extract_notes(file_path):
    y, sr = librosa.load(file_path, sr=None)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

    voiced_pitches = f0[voiced_flag]
    sa = np.median(voiced_pitches)
    
    notes = [get_closest_note(p, sa) for p in f0 if not np.isnan(p)]
    notes = [n for n in notes if n]
    return notes

def identify_raga(note_seq):
    unique_notes = set(note_seq)
    print(f"Detected Notes: {unique_notes}")
    scores = {}
    for raga, info in RAGA_RULES.items():
        score = len(unique_notes & info["notes"])
        scores[raga] = score
    best_match = max(scores.items(), key=lambda x: x[1])
    return best_match[0], best_match[1], scores

if __name__ == "__main__":
    audio_path = "sarang16.wav"  # Replace with your file
    notes = extract_notes(audio_path)
    raga, score, all_scores = identify_raga(notes)

    print(f"\n🎵 Predicted Raga: {raga} (Score: {score})")
    print(f"Raga Scores: {all_scores}")
