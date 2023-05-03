import parse_notes
import prep_seq
import music_lstm

DIRECTORY = "datasets/Debussy_Generic/*.mid"

def train_music_generation_model():
    """
    Train a neural network to generate music.
    """
    
    # Load music notes from MIDI files
    notes = parse_notes.get_notes(DIRECTORY)

    # Determine the number of unique pitch names
    num_pitch_names = len(set(notes))

    # Prepare the input/output sequences for the LSTM network
    input_sequences, output_sequences = prep_seq.prepare_sequences(notes, num_pitch_names)

    # Create the LSTM network model
    lstm_model = music_lstm.create_network(input_sequences, num_pitch_names)

    # Train the LSTM model on the input/output sequences
    music_lstm.train(lstm_model, input_sequences, output_sequences)

if __name__ == '__main__':
    train_music_generation_model()
