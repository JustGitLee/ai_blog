import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.models import load_model
from keras import regularizers

filename = "blog.txt"
with open(filename, encoding='utf-8') as f:
    text = f.read().lower()
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(text)
encoded_text = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index)

seq_length = 100

def data_generator(batch_size):
    X = []
    y = []
    while True:
        for i in range(len(encoded_text) - seq_length):
            X.append(encoded_text[i:i + seq_length])
            y.append(encoded_text[i + seq_length])
            if len(X) == batch_size:
                X_one_hot = to_categorical(X, num_classes=vocab_size + 1)
                yield (X_one_hot, np.array(y))
                X = []
                y = []

embedding_size = 50

model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, vocab_size + 1)))
model.add(Dropout(0.5))
model.add(Dense(vocab_size + 1,
                activation='softmax',
                kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

batch_size=64
steps_per_epoch=len(encoded_text)//batch_size

model.fit(data_generator(batch_size), steps_per_epoch=steps_per_epoch , epochs=10)

model.save('my_model.h5')
# model = load_model('my_model.h5')

generated_text = ""

seed_text = text[:seq_length]

for _ in range(100):
    encoded_seed_text = tokenizer.texts_to_sequences([seed_text])[0]
    encoded_seed_text_one_hot = np.zeros((1, seq_length, vocab_size+2))
    for i in range(seq_length):
        encoded_seed_text_one_hot[0, i, encoded_seed_text[i]] = 1
        prediction_probs = model.predict(encoded_seed_text_one_hot)[0]
        prediction_index = np.argmax(prediction_probs)
        generated_char = tokenizer.sequences_to_texts([[prediction_index + 1]])[0]
        generated_text += generated_char
        seed_text = seed_text[1:] + generated_char

print(generated_text)
