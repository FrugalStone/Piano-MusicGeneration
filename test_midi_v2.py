import mido
import numpy as np

# Load the MIDI file
midi_file = mido.MidiFile('datasets/Debussy_Generic/arabesq1.mid')

# Create an empty array to store the notes
notes = []

# Loop over each track in the MIDI file
for track in midi_file.tracks:
    # Loop over each message in the track
    for msg in track:
        # If the message is a note on message, add it to the array
        if msg.type == 'note_on':
            notes.append(msg)

# Calculate the frequency of each note
freqs = []
for note in notes:
    freq = 440 * 2 ** ((note.note - 69) / 12)
    freqs.append(freq)

# Convert the frequencies to a numpy array
freqs = np.array(freqs)
