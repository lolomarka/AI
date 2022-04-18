import tensorflow as tf
import os

import const

# Name of the checkpoint files
checkpoint_prefix = os.path.join(const.checkpoint_dir, "ckpt_{epoch}")

# рекурентная NN
class RNNMod(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

# модель, предскащывающая следующий символ текста
class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states

# читаем данные из файла
data_path = os.path.join(os.curdir, const.train_data_file)
text = open(data_path, 'rb').read().decode(encoding='utf-8')[:200000]

# создаем словарь символов
vocab = sorted(set(text))

# словарь токен -> ID
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)

# словарь ID -> токен
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

# создаем датасет из текста
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

# делим датасет на последовательности
examples_per_epoch = len(text)//(const.seq_length+1)
sequences = ids_dataset.batch(const.seq_length+1, drop_remainder=True)

# преобразование последовательности в input и label (сдвиг на 1 символ)
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

# применяем split_input_target ко всем последовательностям
dataset = sequences.map(split_input_target)

# формируем датасет для тренировки
dataset = (
    dataset
    .shuffle(const.BUFFER_SIZE)
    .batch(const.BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))


# Length of the vocabulary in chars
vocab_size = len(vocab)

model = RNNMod(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=const.embedding_dim,
    rnn_units=const.rnn_units)

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# тренируем модель
history = model.fit(dataset, epochs=const.epochs,
                    callbacks=[checkpoint_callback])


one_step_model = OneStep(model, chars_from_ids, ids_from_chars)


# тест натренированной модели
def test_model():
    states = None
    next_char = tf.constant(['#include'])
    result = [next_char]

    for _ in range(100):
        next_char, states = one_step_model.generate_one_step(
            next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)

    print(result[0].numpy().decode('utf-8'))


test_model()

# сохраняем модель в файл
tf.saved_model.save(one_step_model, const.save_dir)
