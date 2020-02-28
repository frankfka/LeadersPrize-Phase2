# Paths
ROOT = '/usr/local/'
METADATA_FILEPATH = ROOT + 'dataset/metadata.json'
ARTICLES_FILEPATH = ROOT + 'dataset/articles'
PREDICTIONS_FILEPATH = ROOT + 'predictions.txt'

GENSIM_VECTOR_PATH = './assets/GoogleNewsVectors.bin.gz'
GENSIM_IS_BINARY = True
TRAINED_MODEL_PATH = './assets/model_trained/'

# Prediction params
MAX_SEQ_LEN = 256
BATCH_SIZE = 2
# TODO this is being used in preprocess transformer as well
MAX_CLAIM_LEN = 90
MIN_CLAIM_LEN = 4
MAX_SENT_LEN = 90
MIN_SENT_LEN = 5
MIN_SIMILARITY = 0.5
MAX_SIMILARITY = 1.01

# Pytorch Transformers
MODEL_TYPE_ALBERT = 'albert'
MODEL_TYPE_ROBERTA = 'roberta'
MODEL_TYPE_XLNET = 'xlnet'
MODEL_TYPE_BERT = 'bert'
TORCH_TRANSFORMER_TOKENIZER_PATH = "./assets/albert_tokenizer/"

# BERT
BERT_CONFIG_PATH = './assets/bert_config.json'
BERT_CKPT_PATH = './assets/bert_model.ckpt'
BERT_VOCAB_PATH = './assets/vocab.txt'
BERT_IS_CASED = True

# XLNet
XLNET_TOKENIZER_PATH = "./assets/spiece.model"
XLNET_CONFIG_PATH = "./assets/xlnet_config.json"
XLNET_CKPT_PATH = "./assets/xlnet_model.ckpt"
XLNET_IS_CASED = True

# Raw data DF labels
RAW_CLAIM_ID = 'id'
RAW_CLAIM = 'claim'
RAW_CLAIMANT = 'claimant'
RAW_DATE = 'date'
RAW_LABEL = 'label'
RAW_RELATED_ARTICLES = 'related_articles'
RAW_ARTICLE_ID = 'article_id'
RAW_ARTICLE_TXT = 'text'

# Processed DF labels
CLAIM_ID_IDX = 'claim_id'
TEXT_ONE_IDX = 'text_1'
TEXT_TWO_IDX = 'text_2'
LABEL_IDX = 'label'
