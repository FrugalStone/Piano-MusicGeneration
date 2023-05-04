import mido
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mido import MidiFile
mid = mido.MidiFile('datasets/Debussy_Generic/arabesq1.mid')

# This code reads a MIDI file named 'path_to_midi_file.mid', and for each note-on message, 
# it extracts the pitch, velocity, and frequency of the note.
# The frequency is calculated using the MIDI note-to-frequency conversion formula.

for msg in mid.play():
    if msg.type == 'note_on':
        pitch = msg.note
        velocity = msg.velocity
        frequency = 440 * 2 ** ((pitch - 69) / 12) # MIDI note-to-frequency conversion formula
        print(f"Note: {pitch}, Velocity: {velocity}, Frequency: {frequency}")
