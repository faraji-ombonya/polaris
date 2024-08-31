import os
import re
import numpy as np

from gensim.models import Word2Vec
from transformers import BertTokenizer

from django.conf import settings

from app.models import Book
from app.serializer import BookSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from pgvector.django import CosineDistance

path = os.path.join(settings.BASE_DIR, "word2vec.model")
model = Word2Vec.load(path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class BookListView(APIView):
    def get(self, request, format=None):
        s = request.GET.get("s")
        page = int(request.GET.get("page", 1))
        per_page = int(request.GET.get("per_page", 20))
        distance = float(request.GET.get("distance", 0.5))
        offset = (page - 1) * per_page
        limit = offset + per_page
        books = Book.objects.all()

        if not s:
            response = {
                "count": 0,
                "page": page,
                "per_page": per_page,
                "results": [],
            }
            return Response(response, status=200)

        sentence = s
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
        tokens = tokenizer.tokenize(sentence) 
        valid_tokens = [token for token in tokens if token in model.wv]
        embeddings = np.array([model.wv[token] for token in valid_tokens])
        average_embeddings = (
            np.mean(embeddings, axis=0)
            if embeddings.size > 0
            else np.zeros(model.vector_size)
        )

        # Cosine distance
        books = (
            Book.objects.annotate(
                distance=CosineDistance("embeddings", average_embeddings)
            )
            .order_by("distance")
            .filter(distance__lte=distance)
        )

        # This is a hack that makes sure the whole queryset is evaluated
        # DO NOT DELETE, untill a better solution is found
        len(books)

        count = books.count()
        response = {
            "count": count,
            "page": page,
            "per_page": per_page,
            "results": BookSerializer(books[offset:limit], many=True).data,
        }
        return Response(response, status=status.HTTP_200_OK)
