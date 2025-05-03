#!sudo apt update && sudo apt install tesseract-ocr
#!pip install pytesseract

import os
import cv2
import pandas as pd
import pytesseract

# Set the Tesseract command path for Google Colab
#pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  

# Preprocess an image: Resize and OCR text extraction
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None
    resized_img = cv2.resize(img, (224, 224)) / 255.0  # Normalize
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert for OCR
    text = pytesseract.image_to_string(gray_img).strip()
    return resized_img, text

# Process dataset directory and save to CSV
def process_dataset(image_dir, output_csv, label_map):
    data = {"image_path": [], "text": [], "label": []}
    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            _, text = preprocess_image(image_path)
            if text is not None:
                data["image_path"].append(image_path)
                data["text"].append(text)
                # Infer label from filename
                label = None
                for key, value in label_map.items():
                    if key in filename.lower():
                        label = value
                        break
                data["label"].append(label)

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    label_map = {"real": 1, "fake": 0}  # Update based on naming convention
    image_dir = "/content/drive/MyDrive/Projects/Major_Project/python_files/datasets"  # Path to dataset
    output_csv = "./dataset_with_ocr.csv"
    process_dataset(image_dir, output_csv, label_map)