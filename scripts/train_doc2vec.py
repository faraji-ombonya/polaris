import os
import sys
import re
import time
import multiprocessing
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import BertTokenizer
from tqdm import tqdm


MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(MAIN_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PROJECT.settings")
import django

django.setup()


class Corpus:
    def __init__(self) -> None:
        self.cleaner = re.compile("[^a-zA-Z0-9]")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.path = os.path.join(MAIN_DIR, "BooksDatasetClean.csv")

    def __iter__(self):
        for sentence in tqdm(self.sentences(), desc="Processing sentences"):
            yield sentence

    def sentences(self, chunk_size=50000):
        for chunk in pd.read_csv(self.path, chunksize=chunk_size):
            for index, row in chunk.iterrows():
                title = row["Title"]
                authors = row["Authors"]
                description = row["Description"]
                category = row["Category"]
                publisher = row["Publisher"]
                sentence = f"{title} {authors} {description} {category} {publisher}"
                sentence = self.cleaner.sub(" ", sentence)
                tokenized_sentence = self.tokenizer.tokenize(sentence)
                yield TaggedDocument(tokenized_sentence, [index])


if __name__ == "__main__":
    start_time = time.time()
    corpus_iterable = Corpus()

    model = Doc2Vec(vector_size=100, min_count=4, epochs=5)
    model.build_vocab(corpus_iterable)
    model.train(
        corpus_iterable, 
        total_examples=model.corpus_count, 
        epochs=model.epochs
    )

    model_name = "doc2vec.model"
    path = os.path.join(MAIN_DIR, model_name)
    model.save(path)
    model.wv.save_word2vec_format(f"{path}.txt", binary=False)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Completed in: {duration} seconds.")
