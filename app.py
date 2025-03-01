import os
from datetime import datetime
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
import sqlite3
from pathlib import Path

app = Flask(_name_)
app.secret_key = 'your-secret-key-here' 

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
DB_PATH = 'predictions.db'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         filename TEXT NOT NULL,
         result TEXT NOT NULL,
         confidence REAL NOT NULL,
         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
    ''')
    conn.commit()
    conn.close()

# Load the model at startup
try:
    model = tf.keras.models.load_model('final.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class mapping dictionary
CLASS_MAPPING = {
    0: "implant accepted",
    1: "implant rejected",
    2: "implant disinfected",
    3: "implant infected"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128))  # Adjust size according to your model's requirements
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def save_prediction(filename, result, confidence):
    """Save prediction to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO predictions (filename, result, confidence) VALUES (?, ?, ?)',
              (filename, result, confidence))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/history')
def history():
       conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 50')
    predictions = c.fetchall()
    conn.close()
    return render_template('history.html', predictions=predictions)

@app.route('/predict', methods=['POST'])
def predict():
        if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a PNG or JPEG image.', 'error')
        return redirect(url_for('index'))

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        processed_image = preprocess_image(filepath)
        if processed_image is None:
            flash('Error processing image', 'error')
            return redirect(url_for('index'))

        # Make prediction
        if model is not None:
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            result = CLASS_MAPPING.get(predicted_class, "Unknown")
            
            # Save prediction to database
            save_prediction(filename, result, confidence)
        else:
            result = "Model not loaded"
            flash('Model not loaded', 'error')

        # Clean up - remove uploaded file
        os.remove(filepath)

        return render_template('index.html', result=result, confidence=confidence)

    except Exception as e:
        flash(f'Error processing request: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.before_first_request
def setup():
    init_db()

if _name_ == '_main_':
    app.run(debug=True)



