import pandas as pd
import os
import time


# Read the input into pandas dataframes, returns (json_df, articles_df)
def read_raw_data():
    def content_from_file(file_path):
        with open(file_path, 'r') as file:
            return file.read()

    # JSON Data
    json_df = pd.read_json(METADATA_FILEPATH)
    # Articles
    article_filenames = os.listdir(ARTICLES_FILEPATH)
    article_texts = [content_from_file(os.path.join(ARTICLES_FILEPATH, file_name)) for file_name in article_filenames]
    article_ids = [os.path.splitext(filename)[0] for filename in article_filenames]
    articles_df = pd.DataFrame(data={
        RAW_ARTICLE_ID: article_ids,
        RAW_ARTICLE_TXT: article_texts
    })
    return json_df, articles_df


# Writes result to disk
def write_result(claim_ids, predictions):
    with open(PREDICTIONS_FILEPATH, 'w') as file:
        for claim_id, prediction in zip(claim_ids, predictions):
            file.write(f"{claim_id},{prediction}\n")


def main():
    # Checkpoint
    t = time.time()

    '''
    Read raw input
    '''
    json_df, articles_df = read_raw_data()

    # Checkpoint
    print(f"Raw data loaded in {time.time() - t}s")
    t = time.time()

    '''
    Init vectorizer
    '''
    v = GensimVectorizer(path=GENSIM_VECTOR_PATH, binary=GENSIM_IS_BINARY)

    # Checkpoint
    print(f"Vectorizer loaded in {time.time() - t}s")
    t = time.time()

    '''
    Preprocess info
    '''
    ids, claims, supporting_info = preprocess(json_df, articles_df, vectorizer=v,
                                              max_seq_len=MAX_SEQ_LEN, use_ngrams=True)

    # Checkpoint
    print(f"Preprocessed in {time.time() - t}s")
    t = time.time()

    '''
    Predict from model
    '''
    predictions = predict(claims, supporting_info)

    # Checkpoint
    print(f"Predicted in {time.time() - t}s")
    t = time.time()

    '''
    Write results
    '''
    write_result(ids, predictions)

    # Checkpoint
    print(f"Results written in {time.time() - t}s")


main()
