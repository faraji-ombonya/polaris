from rest_framework import serializers
from app.models import Book


class BookSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = ["id", "title", "author", "description", "category", "publisher"]

    def to_representation(self, instance):
        ret = super().to_representation(instance)

        try:
            ret["distance"] = instance.distance
        except:
            pass

        return ret
