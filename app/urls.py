from django.urls import path
from app import views

urlpatterns = [
    path("v1/", views.BookListView.as_view()),
    path("v2/", views.BookListDocView.as_view()),
]
