# Импортируем необходимые библиотеки
import tensorflow as tf
import numpy as np
import os
import time

# Загружаем датасет Шекспира
file = ('blog.txt')

# Читаем текст из файла и преобразуем его в набор символов
text = open(file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))

# Создаем словарь для кодирования и декодирования символов в числа и обратно
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# Определяем параметры модели: длину входной последовательности, размер пакета данных и размер скрытого состояния RNN
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)
BATCH_SIZE = 64
BUFFER_SIZE = 10000

steps_per_epoch = examples_per_epoch // BATCH_SIZE

embedding_dim = 256

rnn_units = 1024


# Создаем функцию для разделения текста на входные и выходные последовательности для обучения модели
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


# Создаем датасет из тензоров TensorFlow с помощью функции split_input_target
dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
dataset = dataset.batch(seq_length + 1, drop_remainder=True)
dataset = dataset.map(split_input_target)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Создаем модель RNN с помощью Keras Sequential API
model = tf.keras.Sequential([
    # Слой для преобразования чисел в векторы признаков
    tf.keras.layers.Embedding(len(vocab), embedding_dim,
                              batch_input_shape=[BATCH_SIZE, None]),
    # Слой LSTM для обработки последовательных данных
    tf.keras.layers.LSTM(rnn_units,
                         return_sequences=True,
                         stateful=True,
                         recurrent_initializer='glorot_uniform'),
    # Слой для преобразования выхода LSTM в логиты по каждому символу
    tf.keras.layers.Dense(len(vocab))
])


# Определяем функцию потерь как категориальную кросс-энтропию между выходом модели и целевым текстом
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


# Компилируем модель с оптимизатором Adam и функцией потерь loss
model.compile(optimizer='adam', loss=loss)

# Определяем параметры обучения: количество эпох и имя файла для сохранения модели
EPOCHS = 15
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# Обучаем модель
# history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# Определяем параметры ранней остановки
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', # метрика для отслеживания
    patience=2, # количество эпох без улучшения метрики перед остановкой
    restore_best_weights=True # восстанавливаем лучшие веса модели
)

# Обучаем модель с использованием ранней остановки и сохранением точек контроля
history = model.fit(
    dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback, early_stopping]
)

# Загружаем веса модели из последней точки сохранения
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

# model.load_weights(tf.train.latest_checkpoint('./training_checkpoints_copy'))

# Изменяем размер пакета данных на 1
model.build(tf.TensorShape([1, None]))


# Создаем функцию для генерации текста по начальному символу или строке
def generate_text(model, start_string):
    # Количество символов для генерации
    num_generate = 1000

    # Преобразуем начальную строку в числа (векторизация)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Пустой список для хранения сгенерированного текста
    text_generated = []

    # Низкая температура приводит к более предсказуемому тексту.
    # Высокая температура приводит к более неожиданному тексту.
    # Можно экспериментировать с этим параметром.
    temperature = 1.0

    # Сбрасываем состояние модели перед каждой генерацией
    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        # Удаляем размерность пакета данных
        predictions = tf.squeeze(predictions, 0)

        # Используем категориальное распределение для выбора следующего символа
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # Подаем предсказанный символ обратно в модель вместе с предыдущими символами
        input_eval = tf.expand_dims([predicted_id], 0)

        # Добавляем предсказанный символ к сгенерированному тексту
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


# Вызываем функцию и смотрим на результат
print(generate_text(model, start_string = "Соцсети отстой "))