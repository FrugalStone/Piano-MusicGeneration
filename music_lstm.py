import keras.models as kmodel
import keras.layers as klayer

import keras.callbacks as callback

# 12 Pitches - Changes with respect to key
# C C# D D# E F F# G G# A A# B  C
# C Db D Eb E F Gb G Ab A Bb B  C
# 0 1  2 3  4 5 6  7 8  9 10 11 12

# Chords are parsed normal order
# Does not account for inversions / octave displacements

LSTM_UNITS = 512
RECURRENT_DROPOUT_RATE = 0.3
DROPOUT_RATE = 0.3

HIDDEN_LAYER_ACTIVATION_FUNCTION = "relu"
OUTPUT_LAYER_ACTIVATION_FUNCTION = "softmax"

DENSE_UNITS = 256

LOSS_FUNCTION = "categorical_crossentropy"
OPTIMIZER = "rmsprop"


# from keras.layers import LSTM, BatchNormalization, Dropout, Dense, Activation

def create_network(input_data, num_classes):
    """
    Create a neural network model for music generation.
    
    Args:
        input_data: A numpy array of shape (num_samples, sequence_length, num_features)
            representing the input data for the network.
        num_classes: An integer indicating the number of output classes (i.e. unique
            elements in the input data).
    
    Returns:
        A compiled Keras Sequential model representing the neural network.
    """

    # Define the model architecture
    model = kmodel.Sequential()
    model.add(klayer.LSTM(units=LSTM_UNITS, input_shape=(input_data.shape[1], input_data.shape[2]),
                   recurrent_dropout=RECURRENT_DROPOUT_RATE, return_sequences=True))
    model.add(klayer.LSTM(units=LSTM_UNITS, return_sequences=True, recurrent_dropout=RECURRENT_DROPOUT_RATE))
    model.add(klayer.LSTM(units=LSTM_UNITS))
    model.add(klayer.BatchNormalization())
    model.add(klayer.Dropout(rate=DROPOUT_RATE))
    model.add(klayer.Dense(units=DENSE_UNITS))
    model.add(klayer.Activation(HIDDEN_LAYER_ACTIVATION_FUNCTION))
    model.add(klayer.BatchNormalization())
    model.add(klayer.Dropout(rate=DROPOUT_RATE))
    model.add(klayer.Dense(units=num_classes))
    model.add(klayer.Activation(OUTPUT_LAYER_ACTIVATION_FUNCTION))

    # Compile the model
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER)
    
    return model

def train(model, X, y):
    """
    Train a Keras model on input/output data.
    """

    checkpoint_path = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = callback.ModelCheckpoint(
        checkpoint_path,
        monitor='loss',
        save_best_only=True,
        mode='min'
    )

    model.fit(X, y, epochs=200, batch_size=128, callbacks=[checkpoint])