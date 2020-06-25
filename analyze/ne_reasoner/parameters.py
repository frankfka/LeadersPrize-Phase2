
data_path = '../data'

parameters = {
    'model_type': 'esim',
    # 'learning_rate': 0.0004,
    'learning_rate': 0.001,
    'keep_rate': 0.5,
    'seq_length': 50,
    'batch_size': 32,
    'word_embedding_dim': 50,
    'hidden_embedding_dim': 50,
    'embedding_data_path': f'{data_path}/glove.6B.50d.txt',
    'log_path': '../logs',
    'ckpt_path': '../logs'
}
