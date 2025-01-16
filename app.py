from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

# Initialize Flask app
app = Flask(__name__)

# Set model paths
cnn_model_path = "/content/drive/MyDrive/Colab Notebooks/Kidney/CNN/cnn_model_final_complete.keras"
resnet_model_path = "/content/drive/MyDrive/Colab Notebooks/Kidney/Resnet/ResnetFULLYOptimized.keras"
rf_model_path = "/content/drive/MyDrive/Colab Notebooks/Kidney/Randomforst/random_forest_clinical_model.pkl"
xgb_model_path = "/content/drive/MyDrive/Colab Notebooks/Kidney/XGBOOST/optimized_xgb_clinical_model.pkl"

# Load models
cnn_model = load_model(cnn_model_path)
resnet_model = load_model(resnet_model_path)
rf_model = joblib.load(rf_model_path)
xgb_model = joblib.load(xgb_model_path)

# Clinical feature list
clinical_features = ["Age", "BMI", "FamilyHistoryKidneyDisease", "SystolicBP", 
                     "SerumCreatinine", "GFR", "ACR"]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image and clinical data
        image_file = request.files.get('image')
        clinical_data = request.form.to_dict()
        
        # Parse clinical data
        clinical_values = np.array([[float(clinical_data[feature]) for feature in clinical_features]])
        
        # Image processing
        img = Image.open(image_file)
        img = img.resize((224, 224))  # Assuming models expect 224x224 input
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Model selection
        image_model_choice = request.form.get('image_model')  # CNN or ResNet
        clinical_model_choice = request.form.get('clinical_model')  # Random Forest or XGBoost

        # Image prediction
        if image_model_choice == 'cnn':
            image_pred = cnn_model.predict(img_array)
        elif image_model_choice == 'resnet':
            image_pred = resnet_model.predict(img_array)

        # Clinical prediction
        if clinical_model_choice == 'rf':
            clinical_pred = rf_model.predict_proba(clinical_values)[:, 1]
        elif clinical_model_choice == 'xgb':
            clinical_pred = xgb_model.predict_proba(clinical_values)[:, 1]

        # Combine predictions and generate advice
        combined_pred = (image_pred[0][0] + clinical_pred[0]) / 2
        advice = "Signs of kidney disease detected. Consult a nephrologist." if combined_pred > 0.5 else "No major concerns detected."

        return jsonify({
            "image_prediction": float(image_pred[0][0]),
            "clinical_prediction": float(clinical_pred[0]),
            "combined_prediction": float(combined_pred),
            "advice": advice
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
