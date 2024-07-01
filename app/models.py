from django.db import models
from pgvector.django import VectorField

# Create your models here.
class Book(models.Model):
    title = models.TextField(null=True, blank=True)
    author = models.TextField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    publisher = models.TextField(null=True, blank=True)
    embeddings = VectorField(
        dimensions = 100,
        help_text= "Book embeddings",
        null = True,
        blank = True,
    )
    