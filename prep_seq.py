import numpy  # Reshape and normalize input data

import keras.utils as utils

def prepare_sequences(notes, num_pitch_classes):
    """
    Prepare the sequences used by the Neural Network
    """

    # Define the sequence length
    sequence_length = 100

    # Get all unique pitch names in the notes
    pitch_names = sorted(set(note for note in notes))

    # Create a dictionary to map pitch names to integer values
    pitch_to_int = dict((pitch, number) for number, pitch in enumerate(pitch_names))

    # Create input and output sequences for the neural network
    # Every sequence in input_sequences is sequence_length long.
    # Output sequence has the output that should be after input.

    input_sequences = list()
    output_sequence = list()

    # Generate input sequences and their corresponding outputs
    # Run till len(notes) - sequence_length to get every combination
    #   without errors.
    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        input_sequences.append([pitch_to_int[pitch] for pitch in sequence_in])
        output_sequence.append(pitch_to_int[sequence_out])

    num_sequences = len(input_sequences)

    # Reshape the input sequences for use with LSTM layers
    input_sequences = numpy.reshape(input_sequences, (num_sequences, sequence_length, 1))

    # Normalize the input sequences
    input_sequences = input_sequences / float(num_pitch_classes)

    # Convert output sequences to categorical data
    output_sequence = utils.to_categorical(output_sequence)

    return (input_sequences, output_sequence)
