import logging
import os
import sys
import time
import re
import numpy as np


from gensim.models import Word2Vec
from transformers import BertTokenizer

MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(MAIN_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PROJECT.settings")
import django

django.setup()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


import pandas as pd
from app.models import Book


def books():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset_path = os.path.join(MAIN_DIR, "BooksDatasetClean.csv")
    model_path = os.path.join(MAIN_DIR, "word2vec.model")
    model = Word2Vec.load(model_path)

    dataset = (
        pd.read_csv(dataset_path)
        .drop(
            columns=[
                "Price Starting With ($)",
                "Publish Date (Month)",
                "Publish Date (Year)",
            ]
        )
        .fillna("")
        # .sample(100)
    )

    for _, row in dataset.iterrows():
        title = row["Title"]
        authors = row["Authors"]
        description = row["Description"]
        category = row["Category"]
        publisher = row["Publisher"]
        
        sentence = f"{title} {authors} {description} {category} {publisher}"
        sentence = re.sub("[^a-zA-Z0-9]", " ", sentence).strip().lower()
        tokens = [token for token in tokenizer.tokenize(sentence)]
        valid_tokens = [token for token in tokens if token in model.wv]
        embeddings = np.array([model.wv[token] for token in valid_tokens])
        average_embeddings = (
            np.mean(embeddings, axis=0)
            if embeddings.size > 0
            else np.zeros(model.vector_size)
        )

        yield {
            "title": title,
            "author": authors,
            "description": description,
            "category": category,
            "publisher": publisher,
            "embeddings": average_embeddings,
        }


if __name__ == "__main__":
    start_time = time.time()
    Book.objects.bulk_create(
        [Book(**book) for book in books()], ignore_conflicts=True, batch_size=300
    )
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    logger.info(f"Seeding completed in {duration} seconds.")
