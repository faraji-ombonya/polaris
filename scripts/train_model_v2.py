import os
import sys
import re
import time
import multiprocessing
import pandas as pd
from gensim.models import Word2Vec
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
            for _, row in chunk.iterrows():
                title = row["Title"]
                authors = row["Authors"]
                description = row["Description"]
                category = row["Category"]
                publisher = row["Publisher"]
                sentence = f"{title} {authors} {description} {category} {publisher}"
                sentence = self.cleaner.sub(" ", sentence)
                tokenized_sentence = self.tokenizer.tokenize(sentence)
                yield tokenized_sentence


if __name__ == "__main__":
    start_time = time.time()
    corpus_iterable = Corpus()

    w2v = Word2Vec(
        sentences=corpus_iterable,
        vector_size=100,
        window=5,
        workers=multiprocessing.cpu_count(),
        epochs=1,
        min_count=4,
    )

    model_name = "word2vec.model"
    path = os.path.join(MAIN_DIR, model_name)
    w2v.save(path)
    w2v.wv.save_word2vec_format(f"{path}.txt", binary=False)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Completed in: {duration} seconds.")
