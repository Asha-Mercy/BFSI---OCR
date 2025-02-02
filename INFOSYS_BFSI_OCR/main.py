import os
import cv2
import pytesseract
import re
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer

# Flask Configuration
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
CSV_FILE = 'categorized_transactions.csv'
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# MongoDB Configuration
client = MongoClient("mongodb://localhost:27017/")  
db = client['transactions_db']
collection = db['results']

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Set Tesseract OCR Path (Update this path as per your system)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
os.environ["TESSDATA_PREFIX"] = r"C:/Program Files/Tesseract-OCR/tessdata-main"

# Allowed File Extensions
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Load Model and Vectorizer
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, 'rb') as model_file, open(VECTORIZER_PATH, 'rb') as vectorizer_file:
            model = pickle.load(model_file)
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    return None, None

model, vectorizer = load_model()

# Check Allowed File Type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert PDF to Images
def pdf_to_images(pdf_path, output_folder):
    images = convert_from_path(pdf_path, 300)
    saved_images = []
    for page_num, image in enumerate(images, start=1):
        image_path = os.path.join(output_folder, f"page_{page_num}.png")
        image.save(image_path, "PNG")
        saved_images.append(image_path)
    return saved_images

# Preprocess Image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_path = image_path.replace('.png', '_processed.png')
    cv2.imwrite(preprocessed_path, thresholded)
    return preprocessed_path

# Extract Text Using OCR
def extract_text_from_image(image_path):
    return pytesseract.image_to_string(image_path, lang='eng', config='--psm 6')

# Predict Category for a Transaction
def predict_category(description):
    if model and vectorizer:
        desc_tfidf = vectorizer.transform([description])
        return model.predict(desc_tfidf)[0]
    return "Unknown"

# Convert Extracted Text to DataFrame
def process_table_to_dataframe(text):
    lines = text.splitlines()
    headers = ['Transaction Date', 'Description', 'Debit', 'Credit', 'Balance', 'Category']
    rows = []

    for line in lines:
        match = re.match(r'^(\d{2}/\d{2}/\d{4})\s+(.+?)\s+(-?\d+.\d+|-?)\s+(-?\d+.\d+|-?)\s+(-?\d+.\d+|-?)$', line)
        if match:
            row = list(match.groups())
            row.append(predict_category(row[1]))  # Predict Category
            rows.append(row)

    return pd.DataFrame(rows, columns=headers)

# Save DataFrame to CSV
def save_to_csv(df):
    df.to_csv(CSV_FILE, index=False)

# Store Data in MongoDB
def store_dataframe_in_mongo(df):
    collection.insert_many(df.to_dict('records'))

# Routes
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        extracted_text = ""

        if filename.lower().endswith('.pdf'):
            images = pdf_to_images(file_path, app.config['PROCESSED_FOLDER'])
            for img in images:
                preprocessed_img = preprocess_image(img)
                extracted_text += extract_text_from_image(preprocessed_img)
        else:
            preprocessed_img = preprocess_image(file_path)
            extracted_text += extract_text_from_image(preprocessed_img)

        df = process_table_to_dataframe(extracted_text)
        save_to_csv(df)  # Save categorized transactions to CSV
        store_dataframe_in_mongo(df)  # Save to MongoDB

        return redirect(url_for('view_data'))
    return "Invalid file format"

@app.route('/view-data')
def view_data():
    data = list(collection.find({}, {"_id": 0}))  # Fetch all data
    df = pd.DataFrame(data)

    # Visualization: Bar Chart
    fig = px.bar(df, x="Category", y="Debit", title="Spending by Category", color="Category")
    bar_chart = fig.to_html(full_html=False)

    # Pie Chart
    pie_chart = px.pie(df, names="Category", values="Debit", title="Category Distribution").to_html(full_html=False)

    return render_template('view_data.html', data=data, bar_chart=bar_chart, pie_chart=pie_chart)

if __name__ == '__main__':
    app.run(debug=True)
