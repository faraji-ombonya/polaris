from django.db import models
from pgvector.django import VectorField, HnswIndex


# Create your models here.
class Book(models.Model):
    title = models.TextField(null=True, blank=True)
    author = models.TextField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    publisher = models.TextField(null=True, blank=True)
    category = models.TextField(null=True, blank=True)
    embeddings = VectorField(
        dimensions = 100,
        help_text= "Book embeddings",
        null = True,
        blank = True,
    )
    
    class Meta:
        indexes = [
            HnswIndex(
                name="embeddings_index",
                fields = ["embeddings"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            )
        ]