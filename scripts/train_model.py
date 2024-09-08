import os
import sys
import re
import time
import pandas as pd
from gensim.models import Word2Vec
from transformers import BertTokenizer


MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(MAIN_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PROJECT.settings")
import django

django.setup()


if __name__ == "__main__":
    # read dataset
    start_time = time.time()
    file_name = "ml-parts.csv"
    path = os.path.join(MAIN_DIR, "BooksDatasetClean.csv")
    dataset = pd.read_csv(path).fillna(" ")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # basic processing
    texts = []
    for index, row in dataset.iterrows():
        title = row["Title"]
        authors = row["Authors"]
        description = row["Description"]
        category = row["Category"]
        publisher = row["Publisher"]
        text = f"{title} {authors} {description} {category} {publisher}"
        text = re.sub("[^a-zA-Z0-9]", " ", text)
        texts.append(text)

    sentences = [tokenizer.tokenize(line) for line in texts]

    w2v = Word2Vec(
        sentences, vector_size=100, window=5, workers=3, epochs=1, min_count=4
    )
    model_name = "word2vec.model"
    path = os.path.join(MAIN_DIR, model_name)
    w2v.save(path)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Completed in: {duration} seconds.")
