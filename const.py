# Путь к датасету
train_data_file = 'data.txt'

seq_length = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000
epochs = 30

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'

# Путь к модели
save_dir = 'one_step_model'