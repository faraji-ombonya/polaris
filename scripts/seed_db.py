import logging
import os
import sys
import time

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


if __name__ == "__main__":
    start_time = time.time()
    Book.objects.all().delete()
    path = os.path.join(MAIN_DIR, "BooksDatasetClean.csv")

    dataset = (
        pd.read_csv(path)
        .sample(n=10000)
        .drop(
            columns=[
                "Price Starting With ($)",
                "Publish Date (Month)",
                "Publish Date (Year)",
            ]
        )
        .fillna("")
    )

    books = []
    for index, row in dataset.iterrows():
        books.append(
            {
                "title": row["Title"],
                "author": row["Authors"],
                "description": row["Description"],
                "category": row["Category"],
                "publisher": row["Publisher"],
            }
        )

    Book.objects.bulk_create(
        [Book(**book) for book in books], ignore_conflicts=True, batch_size=300
    )
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    logger.info(f"Seeding completed in {duration} seconds.")
