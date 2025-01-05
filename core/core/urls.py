"""
URL configuration for core project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from home.views import *

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", index, name = 'home'),
    path('linear/', linear, name = 'linear'),
    path('logistic/', logistic_regression, name = 'logistic'),
    path('select/',select, name = 'select'),
    path('svm/', svm_classification, name = 'svm'),
    path('decision-tree/', decision_tree, name = 'decision_tree'),
    path('download-model/<str:filename>/', download_model, name='download_model'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
