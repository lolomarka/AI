import tensorflow as tf
import const
import time


# загружаем натренированную модель
one_step_model = tf.saved_model.load(const.save_dir)

start = time.time()
states = None

# начальное состояние текста
next_char = tf.constant(['fn '])
result = [next_char]

# генерация 1000 символов текста
for n in range(1000):
    next_char, states = one_step_model.generate_one_step(
        next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)
