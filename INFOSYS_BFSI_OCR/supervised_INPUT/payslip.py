import os
import cv2
import pytesseract
import pdf2image
import pandas as pd
import re
import pymongo

# Tesseract OCR configuration
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
os.environ["TESSDATA_PREFIX"] = r"C:/Program Files/Tesseract-OCR/tessdata"

# Define file paths
input_pdf_path = r"C:/Users/Asha Mercy R/OneDrive/Desktop/missingdata/payslip2.pdf"
temp_image_folder = "temp_images"
os.makedirs(temp_image_folder, exist_ok=True)

# Step 1: Convert PDF to PNG images
print("Converting PDF to images...")
images = pdf2image.convert_from_path(input_pdf_path)
image_paths = []

for i, image in enumerate(images):
    image_path = os.path.join(temp_image_folder, f"page_{i+1}.png")
    image.save(image_path, "PNG")
    image_paths.append(image_path)

# Step 2: Preprocess each image, save it, and extract text
def preprocess_image(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.GaussianBlur(thresh, (5, 5), 0)
    
    # Save the processed image
    cv2.imwrite(output_path, denoised)
    
    return denoised

data = []

print("Preprocessing images and extracting text...")
for i, image_path in enumerate(image_paths):
    # Define the path for the preprocessed image
    processed_image_path = os.path.join(temp_image_folder, f"processed_page_{i+1}.png")
    
    # Preprocess and save the image
    preprocessed_image = preprocess_image(image_path, processed_image_path)
    
    # Extract text using the preprocessed image
    extracted_text = pytesseract.image_to_string(preprocessed_image, lang='eng', config='--psm 6')
    
    # Debugging: Print extracted text
    print(f"Extracted text from page {i+1}:")
    print(extracted_text)  # This helps you verify the text structure
    
    # Regex patterns for Earnings and Deductions
    earnings_pattern = r"Basic Salary[\s\S]+?Performance Bonus [\d,\.]+"
    deductions_pattern = r"PF Contribution[\s\S]+?Soft Loan [\d,\.]+"

    # Extract Earnings section
    earnings_match = re.search(earnings_pattern, extracted_text)
    if earnings_match:
        earnings_lines = earnings_match.group(0).splitlines()
        for line in earnings_lines:
            match = re.match(r"(.+?)\s+([\d,]+\.\d{2})\s+[\d,]+\.\d{2}", line)
            if match:
                description = match.group(1).strip()
                amount = match.group(2).replace(",", "")
                data.append(["Earnings", description, amount])
    
    # Extract Deductions section
    deductions_match = re.search(deductions_pattern, extracted_text)
    if deductions_match:
        deductions_lines = deductions_match.group(0).splitlines()
        for line in deductions_lines:
            # Match lines that have the structure Description | Current Period Amount | Year to Date Amount
            match = re.match(r"(.+?)\s*\|\s*([\d,]+\.\d{2})\s*\|\s*([\d,]+\.\d{2})", line)
            if match:
                description = match.group(1).strip()
                amount = match.group(2).replace(",", "")  # Use current period amount
                data.append(["Deductions", description, amount])

# Step 3: Save extracted data to CSV
csv_output_path = "payslip_data_corrected.csv"
df = pd.DataFrame(data, columns=["Type", "Description", "Amount"])
df.to_csv(csv_output_path, index=False)

print(f"Extraction complete. Data saved to {csv_output_path}")

# Clean up temporary images
for image_path in image_paths:
    os.remove(image_path)

print("Temporary images cleaned up.")

# Step 4: Store extracted data into MongoDB
# MongoDB connection setup
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string if different
db_name = "PayslipDB"  # Name of the database
collection_name = "PayslipData"  # Name of the collection
db = mongo_client[db_name]
collection = db[collection_name]

# Load data from CSV
data = pd.read_csv(csv_output_path)

# Convert the DataFrame to a list of dictionaries for MongoDB
records = data.to_dict(orient='records')

# Insert the records into MongoDB
try:
    result = collection.insert_many(records)
    print(f"Data successfully inserted into MongoDB. Inserted IDs: {result.inserted_ids}")
except Exception as e:
    print(f"An error occurred: {e}")

# Verify the inserted data
print(f"Total records in MongoDB collection '{collection_name}': {collection.count_documents({})}")
