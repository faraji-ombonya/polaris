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

if __name__ == "__main__":
    path = os.path.join(MAIN_DIR, "word2vec.model")
    model = Word2Vec.load(path)
    print(model.wv.most_similar("book"))

    for book in Book.objects.all():
        sentence = f"{book.title} {book.description} {book.author} {book.category}"    
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        words = [word for word in sentence.split()]

        print(words)

        # valid_words = [vword for vword in words if vword in model.wv]

        # valid_words_embeddings = np.array([model.wv[word] for word in valid_words])

        # if valid_words_embeddings.size > 0:
        #     averaged_embeddings = np.mean(valid_words_embeddings, axis=0)
        # else:
        #     averaged_embeddings = np.zeros(model.vector_size)

        # book.embeddings = averaged_embeddings
        # book.save()

        # print(averaged_embeddings)    
