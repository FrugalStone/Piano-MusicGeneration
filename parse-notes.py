import glob   # Retrieve files
import pickle # Serialize notes
import music21

def get_notes(directory):

    """
    Store individual notes and chords in a list
    And write it to a file at last
    """

    notes = list()

    # Process Midi file in the given directory
    # Get all notes and chords
    for file in glob.glob(directory):
        print("Parsing file %s" % file)
        parsed_midi = music21.converter.parse(file)

        try:
            # Midi file has more than one instrument
            # Instrument at index 0 probably has melody line
            instrument_parts = music21.instrument.partitionByInstrument(parsed_midi)
            notes_by_instrument = instrument_parts.parts[0].recurse()
        except:
            # Midi file is monophonic
            notes_by_instrument = parsed_midi.flat.notes
    
        # Convert notes / chords to string and append to notes
        for structure in notes_by_instrument:
            if isinstance(structure, music21.note.Note):
                notes.append(str(structure.pitch))
            elif isinstance(structure, music21.chord.Chord):
                # Return notes in normal order
                # Normal order representation of C Major triad - [0, 4, 7]
                notes.append(".".join(map(str, structure.normalOrder)))

    # Open a file called 'notes' in write binary mode 
    # and save the notes data to it using pickle
    with open('data/notes', 'wb') as file:
        pickle.dump(notes, file)

    return notes
