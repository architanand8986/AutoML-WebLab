from django.apps import AppConfig


class HomeConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "home"
    def ready(self):
        from .utils import cleanup_old_models
        cleanup_old_models()
