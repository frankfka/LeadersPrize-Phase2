import pickle


def load_dictionary(save_path: str):
    with open(save_path, 'rb') as f:
        dictionary = pickle.load(f)
        return dictionary

