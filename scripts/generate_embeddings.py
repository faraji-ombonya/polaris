import os
import sys
import re

MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(MAIN_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PROJECT.settings")
import django

django.setup()

import pandas as pd
import numpy as np
from app.models import Book
from gensim.models import Word2Vec
from transformers import BertTokenizer

if __name__ == "__main__":
    path = os.path.join(MAIN_DIR, "word2vec.model")
    model = Word2Vec.load(path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for book in Book.objects.all():
        sentence = f"{book.title} {book.description} {book.author} {book.category} {book.publisher}"
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
        tokens = tokenizer.tokenize(sentence)
        valid_tokens = [token for token in tokens if token in model.wv]
        embeddings = np.array([model.wv[token] for token in valid_tokens])
        average_embeddings = (
            np.mean(embeddings, axis=0)
            if embeddings.size > 0
            else np.zeros(model.vector_size)
        )

        book.embeddings = average_embeddings
        book.save()
