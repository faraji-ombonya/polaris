import os
import re
import numpy as np

from gensim.models import Word2Vec

from django.conf import settings

from app.models import Book
from app.serializer import BookSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from pgvector.django import CosineDistance
from app.models import Book

class BookListView(APIView):
    def get(self, request, format=None):
        books = Book.objects.all()

        s = request.GET.get("s")

        if s:

            path = os.path.join(settings.BASE_DIR, "word2vec.model")
            model = Word2Vec.load(path)
            print(model.wv.most_similar("book"))

            sentence = s  
            sentence = re.sub('[^a-zA-Z]', ' ', sentence)

            words = [word for word in sentence.split()]

            valid_words = [vword for vword in words if vword in model.wv]

            valid_words_embeddings = np.array([model.wv[word] for word in valid_words])

            if valid_words_embeddings.size > 0:
                averaged_embeddings = np.mean(valid_words_embeddings, axis=0)
            else:
                averaged_embeddings = np.zeros(model.vector_size)

            books = books.annotate(
                distance=CosineDistance("embeddings", averaged_embeddings)
            ).order_by("distance")[:5]
        else:
            books = books[:2]


        serializer = BookSerializer(books, many=True)
        return Response(serializer.data)

