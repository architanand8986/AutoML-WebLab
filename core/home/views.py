from django.shortcuts import render, redirect
from .utils import *
from .models import *
from django.http import FileResponse, HttpResponse
from django.conf import settings
import os

# Create your views here.




def download_model(request, filename):
    try:
        # Construct full path to model file
        file_path = os.path.join(settings.MEDIA_ROOT, 'models', filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return HttpResponse("Model file not found", status=404)
            
        # Check file extension
        if not filename.endswith('.pkl'):
            return HttpResponse("Invalid file type", status=400)
            
        # Open and return file
        model_file = open(file_path, 'rb')
        response = FileResponse(model_file, content_type='application/octet-stream')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
        
    except Exception as e:
        return HttpResponse(f"Error downloading model: {str(e)}", status=500)

def select(request):
    return render(request, "select.html")

def index(request):
    context = {'page': 'Home page'}
    if request.method == 'POST':
        dataset = request.FILES['dataset']

        UploadedFile.objects.all().delete()

        uploaded_file = UploadedFile(file=dataset)
        uploaded_file.save() 
        return redirect('select')
    return render(request, "index.html", context)

def linear(request):
    try:
        latest_file = UploadedFile.objects.latest('id')  # Use the 'id' field for ordering
        file_path = latest_file.file.path
        result = predict_with_linear_regression(file_path)
        result.update({'page': 'Result page'})# delete the file after use
        return render(request, "linear.html", result)
    except Exception as e:
        return render(request, "error.html", {'error_message': str(e)})

def logistic_regression(request):
    try:
        latest_file = UploadedFile.objects.latest('id')  # Use the 'id' field for ordering
        file_path = latest_file.file.path
        result = predict_with_logistic_regression(file_path)
        result.update({'page': 'Logistic Regression'})
        return render(request, "logistic.html", result)
    except Exception as e:
        return render(request, "error.html", {'error_message': str(e)})

def svm_classification(request):
    try:
        latest_file = UploadedFile.objects.latest('id')
        file_path = latest_file.file.path
        result = predict_with_svm(file_path)
        result.update({'page': 'SVM Classification'})
        return render(request, "svm.html", result)
    except Exception as e:
        return render(request, "error.html", {'error_message': str(e)})

def decision_tree(request):
    try:
        latest_file = UploadedFile.objects.latest('id')
        file_path = latest_file.file.path
        result = predict_with_decision_tree(file_path)
        result.update({'page': 'Decision Tree Classification'})
        return render(request, "decision_tree.html", result)
    except Exception as e:
        return render(request, "error.html", {'error_message': str(e)})