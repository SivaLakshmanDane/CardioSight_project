from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
import pickle  # <-- added for loading heart model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import tensorflow as tf
from PIL import Image
import io
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'MOBILENET.h5'
PLOTS_FOLDER = 'static/plots'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the hypertensive retinopathy model
model = load_model(MODEL_PATH)
class_names = ['Hypertensive', 'Normal']

# Load the heart disease model (pickle file)
HEART_MODEL_PATH = 'models/heart_model.pkl'
with open(HEART_MODEL_PATH, 'rb') as f:
    heart_model = pickle.load(f)

# Feature order expected by the heart model (from reference project)
heart_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                  'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array


def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64


@app.route('/', methods=['GET'])
def home():
    # For GET request, just show the combined input form
    plot_files = [f for f in os.listdir(PLOTS_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return render_template('index.html', plot_files=plot_files)


@app.route('/predict_combined', methods=['POST'])
def predict_combined():
    # --- Validate and save retinal image ---
    if 'file' not in request.files:
        flash('No retinal image selected', 'error')
        return redirect(url_for('home'))
    file = request.files['file']
    if file.filename == '':
        flash('No retinal image selected', 'error')
        return redirect(url_for('home'))
    if not allowed_file(file.filename):
        flash('Allowed image types: png, jpg, jpeg', 'error')
        return redirect(url_for('home'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # --- Hypertensive Retinopathy Prediction ---
    img_array = preprocess_image(filepath)
    pred = model.predict(img_array)
    hypertensive_index = np.argmax(pred)
    hypertensive_class = class_names[hypertensive_index]
    hypertensive_confidence = float(np.max(pred)) * 100
    hypertensive_positive = (hypertensive_class == 'Hypertensive')

    # --- Heart Disease Prediction ---
    try:
        heart_input = []
        for feature in heart_features:
            value = request.form.get(feature)
            if value is None or value.strip() == '':
                flash(f'Missing value for {feature}', 'error')
                return redirect(url_for('home'))
            heart_input.append(float(value))
        heart_array = np.array(heart_input).reshape(1, -1)
        heart_pred = heart_model.predict(heart_array)[0]
        heart_positive = bool(heart_pred)
    except Exception as e:
        flash(f'Error processing heart disease data: {str(e)}', 'error')
        return redirect(url_for('home'))

    print("Results",hypertensive_positive,heart_positive)
    
    # --- Determine combined stage ---
    if hypertensive_positive and heart_positive:
        stage = "Stage 2"
        stage_description = "Both Hypertensive Retinopathy and Heart Disease Risk Detected"
    elif hypertensive_positive:
        stage = "Stage 1 (Hypertensive Retinopathy)"
        stage_description = "Hypertensive Retinopathy Detected"
    elif heart_positive:
        stage = "Stage 1 (Heart Risk)"
        stage_description = "Heart Disease Risk Detected, but no Hypertensive Retinopathy"
    else:
        stage = "Normal"
        stage_description = "No significant risk detected"

    # --- Create visualisation of the retinal image ---
    img = Image.open(filepath)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Retinal Image\nPrediction: {hypertensive_class}\nConfidence: {hypertensive_confidence:.2f}%',
              pad=20, fontsize=14, color='green' if hypertensive_class == 'Normal' else 'red')
    plot_url = plot_to_base64()

    # List existing model performance plots (optional)
    plot_files = [f for f in os.listdir(PLOTS_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]

    return render_template('result.html',
                           hypertensive_class=hypertensive_class,
                           hypertensive_confidence=hypertensive_confidence,
                           heart_prediction='High Risk' if heart_positive else 'Low Risk',
                           stage=stage,
                           stage_description=stage_description,
                           uploaded_image=filename,
                           plot_url=plot_url,
                           plot_files=plot_files)


if __name__ == '__main__':
    app.run(debug=True)