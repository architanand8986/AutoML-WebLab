# AI/ML Model Deployment with Django  

This project combines the power of **Django** for backend hosting and **Bootstrap** for a responsive frontend, creating a seamless platform for deploying and interacting with AI/ML models. Users can upload datasets, train machine learning models, evaluate their performance, and download trained models for future use.

## **Key Features**
- **AI/ML Functionality**:  
  - Supports training and evaluation of models such as:
    - Linear Regression
    - Logistic Regression
    - Decision Tree
    - Support Vector Machine (SVM)
  - Real-time metrics like accuracy, mean squared error, and classification reports.
  - Downloadable trained models for future applications.
  
- **Django Backend**:  
  - Handles file uploads, processes datasets, and trains AI/ML models.
  - Provides routes for interactive user input and dynamic results.
  
- **Bootstrap Frontend**:  
  - Clean and user-friendly interface for uploading datasets and selecting models.
  - Dropdown options for model selection and real-time form validation.

## **How It Works**
1. **Dataset Upload**:  
   Users upload a CSV dataset via the web interface.
   
2. **Model Training**:  
   Select a machine learning model (e.g., Linear Regression) and train it using the uploaded dataset.
   
3. **Performance Evaluation**:  
   View model evaluation metrics such as confusion matrices, accuracy scores, and regression errors.

4. **Model Download**:  
   Download the trained model in `.pkl` format for reuse.

5. **Automatic Cleanup**:  
   Trained models are automatically deleted after download to save server space.

## **Tech Stack**
- **Backend**: Django (Python-based web framework)  
- **Frontend**: Bootstrap (CSS framework for responsiveness)  
- **Machine Learning**: Scikit-learn (for model implementation and evaluation)  
- **Data Visualization**: Matplotlib and Plotly  
- **File Handling**: Django's media file management system.

## **Setup Instructions**
1. Clone the repository:  
   ```bash
   git clone https://github.com/architanand8986/AI-ML-Model-Deployment.git
2. Navigate to the project directory
   ```bash
   cd AI-ML-Model-Deployment

3. Install dependencies
   ```bash
   pip install -r requirements.txt

4. Run migrations
   ```bash
    python manage.py makemigrations
    python manage.py migrate

5. Start the development server
   ```bash
    python manage.py runserver

