import requests
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def get_audio_features(track_id, access_token):
    track_info_url = f'https://api.spotify.com/v1/audio-features/{track_id}'
    headers = {
        'Authorization': f'Bearer {access_token}',
    }
    track_info_response = requests.get(track_info_url, headers=headers)
    track_info = track_info_response.json()
    return track_info

def compute_chromagram(preview_url):
    if preview_url is None:
        raise ValueError("Preview URL is missing or invalid")

    # Load audio data from the preview URL
    try:
        y, sr = librosa.load(preview_url)
    except Exception as e:
        raise IOError(f"Error loading audio from preview URL: {e}")

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return chroma

def analyze_audio_features(track_info):
    energy = track_info['energy']
    valence = track_info['valence']

    print(f"The energy of this track is: {energy}")
    if energy > 0.5:
        print("This track is high-energy.")
    else:
        print("This track is low-energy.")

    print(f"The valence of this track is: {valence}")

if __name__ == '__main__':
    client_id = 'a01297bedc014ad7850bd4bae413e41f'
    client_secret = 'a3f045d5c610453baf84e45d854e7b87'

    auth_url = 'https://accounts.spotify.com/api/token'
    auth_data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
    }

    auth_response = requests.post(auth_url, data=auth_data)
    auth_response_data = auth_response.json()
    access_token = auth_response_data['access_token']

    track_id = input("Enter the track ID: ")

    track_info = get_audio_features(track_id, access_token)
    analyze_audio_features(track_info)

    # Get preview URL of the track
    track_info_url = f'https://api.spotify.com/v1/tracks/{track_id}'
    headers = {
        'Authorization': f'Bearer {access_token}',
    }
    track_info_response = requests.get(track_info_url, headers=headers)
    track_info = track_info_response.json()
    preview_url = track_info['preview_url']

    # Compute and print dominant notes or chords
    try:
        chroma = compute_chromagram(preview_url)
        print("Dominant notes or chords:")
        for i, frame in enumerate(chroma.T):
            max_idx = np.argmax(frame)
            note_name = librosa.core.midi_to_note(max_idx)
            print(f"Frame {i}: Dominant note or chord: {note_name}")
    except ValueError as ve:
        print(ve)
    except IOError as ioe:
        print(ioe)

chord_dict = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B'
}


def get_chord_name(chroma_bin):
    # Find the index of the maximum value in the chroma bin
    max_idx = np.argmax(chroma_bin)

    # Map the index to the corresponding chord name using the chord dictionary
    chord_name = chord_dict[max_idx]

    return chord_name


# Example usage:
# Suppose 'chroma_bin' is a chroma vector obtained from librosa.feature.chroma_stft()

# Get the chord name from the chroma bin
chord_name = get_chord_name(chroma_bin)

print("Detected Chord:", chord_name)
