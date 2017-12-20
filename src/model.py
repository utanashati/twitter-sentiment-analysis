from keras import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

from utils import ALPHABET, BATCHSIZE, MAX_WORDS_TWEET, MAX_CHARS_WORD

RECEPTIVE_FIELDS = [1, 2, 3, 4, 5, 6]
OUTPUT_FILTERS = [25, 50, 75, 100, 125, 150]

def compile_CharLSTM():
    model = Sequential()

    model.add(Conv2D(OUTPUT_FILTERS[0], (1, RECEPTIVE_FIELDS[0]), activation='relu', \
                     input_shape=(MAX_WORDS_TWEET, MAX_CHARS_WORD, len(ALPHABET))))
    print('Conv2D 1: ' + str(model.output_shape))
    for output_filters, receptive_field in zip(OUTPUT_FILTERS[1:], RECEPTIVE_FIELDS[1:]):
        model.add(Conv2D(output_filters, (1, receptive_field), activation='relu'))
        print('Conv2D {}: '.format(receptive_field) + str(model.output_shape))

    model.add(MaxPooling2D((1, 2), data_format='channels_first'))
    print('MaxPooling2D: ' + str(model.output_shape))
    model.add(Reshape((model.output_shape[1], model.output_shape[3])))
    print('Reshape: ' + str(model.output_shape))

    model.add(LSTM(units=650))
    print('LSTM: ' + str(model.output_shape))

    model.add(Dense(1, activation='sigmoid'))
    print('Dense: ' + str(model.output_shape))

    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model