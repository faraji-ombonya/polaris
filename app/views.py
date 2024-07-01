from app.models import Book
from app.serializer import BookSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


class BookListView(APIView):
    def get(self, request, format=None):
        books = Book.objects.all()[:100]
        serializer = BookSerializer(books, many=True)
        return Response(serializer.data)

