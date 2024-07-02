import os
import sys

MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(MAIN_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PROJECT.settings")
import django
django.setup()

import pandas as pd
from app.models import Book


if __name__ == "__main__":
    path = os.path.join(MAIN_DIR, "BooksDatasetClean.csv")
    dataset = pd.read_csv(path)

    dataset = dataset.sample(n=10000)

    dataset = dataset.drop(
        columns=[
            "Price Starting With ($)", 
            "Publish Date (Month)",
            "Publish Date (Year)",
            ]
        )

    dataset = dataset.fillna("")


    # print(dataset.head())

    for i in range(0, len(dataset)):

        title = dataset["Title"][i].strip()
        author = dataset["Authors"][i].strip()
        description = dataset["Description"][i].strip()
        category = dataset["Category"][i].strip()
        publisher = dataset["Publisher"][i].strip()

        try:
            Book.objects.create(
                title=title,
                author=author,
                description=description,
                category=category,
                publisher = publisher,
            )
        except Exception as e:
            print("An error occured.")
            print(e)
