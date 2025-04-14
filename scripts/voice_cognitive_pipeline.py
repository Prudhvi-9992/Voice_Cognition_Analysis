import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd

def convert_mp3_to_wav(mp3_path, wav_path):
    try:
        audio, sr = librosa.load(mp3_path, sr=None)
        sf.write(wav_path, audio, sr)
    except Exception as e:
        print(f"[ERROR] Failed to convert {mp3_path} to WAV: {e}")

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        features = {
            'filename': os.path.basename(file_path),
            'duration_sec': librosa.get_duration(y=y, sr=sr),
            'tempo': librosa.beat.tempo(y=y, sr=sr)[0],
            'zcr_mean': np.mean(librosa.feature.zero_crossing_rate(y)[0]),
            'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]),
            'spectral_rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]),
            'rmse_mean': np.mean(librosa.feature.rms(y=y)[0]),
            'mfcc_1_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[0])
        }
        return features

    except Exception as e:
        print(f"[ERROR] Failed to extract features from {file_path}: {e}")
        return None

def process_all_audios(audio_dir):
    data = []
    wav_dir = os.path.join(audio_dir, "converted_wav")
    os.makedirs(wav_dir, exist_ok=True)

    for filename in os.listdir(audio_dir):
        if filename.lower().endswith(".mp3"):
            mp3_path = os.path.join(audio_dir, filename)
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            wav_path = os.path.join(wav_dir, wav_filename)

            convert_mp3_to_wav(mp3_path, wav_path)
            features = extract_features(wav_path)
            if features:
                data.append(features)

    return pd.DataFrame(data)

def main():
    audio_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "audio_samples")
    df = process_all_audios(audio_folder)

    if not df.empty:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cognitive_features.csv")
        df.to_csv(csv_path, index=False)
        print(f"[SUCCESS] Feature extraction complete. Results saved to {csv_path}")
    else:
        print("[INFO] No audio features extracted.")

if __name__ == "__main__":
    main()
