from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from django.conf import settings
import pickle
import uuid
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
os.makedirs(MEDIA_ROOT, exist_ok=True)

def cleanup_old_models():
    """Delete models older than 24 hours"""
    models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
    if os.path.exists(models_dir):
        current_time = datetime.datetime.now()
        for filename in os.listdir(models_dir):
            file_path = os.path.join(models_dir, filename)
            file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            if (current_time - file_modified).total_seconds() > 21600:  # 6 hours
                os.remove(file_path)

def save_model_pickle(model, model_type):
    """Helper function to save model"""
    cleanup_old_models()
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{model_type}_{unique_id}.pkl"
    model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return filename


def predict_with_linear_regression(file_path):
    data = pd.read_csv(file_path)

    if data.shape[1] < 2:
        raise ValueError("Dataset must have at least one feature column and one target column.")

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    model_filename = save_model_pickle(model, 'linear')

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    fig = px.scatter(
        x=y_test,
        y=predictions,
        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
        title="Actual vs Predicted Values"
    )
    fig.add_shape( 
        type="line",
        x0=min(y_test),
        y0=min(y_test),
        x1=max(y_test),
        y1=max(y_test),
        line=dict(color="red", dash="dash")
    )
    graph_html = fig.to_html(full_html=False)

    return {
        "predictions": predictions.tolist(),
        "mse": mse,
        "visualization": graph_html,
        "model_filename": model_filename
    }

def predict_with_logistic_regression(file_path):

    data = pd.read_csv(file_path)
    print("Dataset loaded successfully")
    print(data.head())
    
    if data.shape[1] < 2:
        raise ValueError("Dataset must have at least one feature column and one target column.")

    X = data.iloc[:, :-1]  
    y = data.iloc[:, -1]   
    print("Features and target split successfully")
    print("Target variable unique values:", y.unique())

    if y.nunique() != 2:
        raise ValueError("Target variable must be binary for logistic regression.")
    
    if not X.select_dtypes(include=["number"]).shape[1] == X.shape[1]:
        X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train-test split done")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    model_filename = save_model_pickle(model, 'logistic')
    print("Model trained successfully")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot to file using settings.MEDIA_ROOT
    plot_filename = 'confusion_matrix.png'
    plot_path = os.path.join(settings.MEDIA_ROOT, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return {
        "predictions": predictions.tolist(),
        "accuracy": accuracy,
        "classification_report": report,
        "plot_path": settings.MEDIA_URL + plot_filename,
        "model_filename": model_filename
    }

def predict_with_svm(file_path):

    data = pd.read_csv(file_path)
    
    if data.shape[1] < 2:
        raise ValueError("Dataset must have at least one feature column and one target column.")

    X = data.iloc[:, :-1]  
    y = data.iloc[:, -1]   
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    model_filename = save_model_pickle(model, 'svm')

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('SVM Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_filename = 'svm_confusion_matrix.png'
    plot_path = os.path.join(settings.MEDIA_ROOT, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return {
        "predictions": predictions.tolist(),
        "accuracy": accuracy,
        "classification_report": report,
        "plot_path": settings.MEDIA_URL + plot_filename,
        "model_filename": model_filename
    }


def predict_with_decision_tree(file_path):
    data = pd.read_csv(file_path)
    
    if data.shape[1] < 2:
        raise ValueError("Dataset must have at least one feature column and one target column.")

    X = data.iloc[:, :-1]  
    y = data.iloc[:, -1]   
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    model_filename = save_model_pickle(model, 'decision_tree')
    

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Decision Tree Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_filename = 'decision_tree_confusion_matrix.png'
    plot_path = os.path.join(settings.MEDIA_ROOT, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return {
        "predictions": predictions.tolist(),
        "accuracy": accuracy,
        "classification_report": report,
        "plot_path": settings.MEDIA_URL + plot_filename,
        "model_filename": model_filename
    }