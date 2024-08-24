from django.urls import path
from . import views

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('get_all/', views.get_transcript_long, name = 'get_transcript_long'),
    path('get_summary/', views.get_summary, name='get_summary'),
    path('get_transcript/', views.get_transcript, name='get_transcript'),
    path('get_similarity/', views.get_score, name='get_similarity'),
]
