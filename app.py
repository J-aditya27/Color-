from random import choice
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from PIL import Image
import io
import base64
import joblib
from mtcnn import MTCNN 
import xgboost as xgb
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from skincare_recommendations import skincare_suggestions

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# Load the trained model
model = joblib.load('ml/recommendation_model.joblib')

app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize MTCNN face detector
detector = MTCNN()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Color palettes with associated hex values
color_palettes = {
    "Pastels": [("#FFB6C1", "Pastel Pink"), ("#D8BFD8", "Thistle"), ("#D3D3D3", "Light Grey"), ("#FFC0CB", "Pink"), ("#B0E0E6", "Powder Blue")],
    "Earthy Tones": [("#8B4513", "Saddle Brown"), ("#A52A2A", "Brown"), ("#D2691E", "Chocolate"), ("#F4A300", "Vivid Orange"), ("#7C4700", "Dark Orange")],
    "Warm Tones": [("#FF4500", "Orange Red"), ("#FF6347", "Tomato"), ("#FFD700", "Gold"), ("#FF8C00", "Dark Orange"), ("#F0E68C", "Khaki")],
    "Neutrals": [("#808080", "Grey"), ("#A9A9A9", "Dark Grey"), ("#D3D3D3", "Light Grey"), ("#F5F5F5", "White Smoke"), ("#F0F0F0", "Very Light Gray")],
    "Cool Tones": [("#4682B4", "Steel Blue"), ("#5F9EA0", "Cadet Blue"), ("#00CED1", "Dark Turquoise"), ("#7B68EE", "Medium Slate Blue"), ("#B0E0E6", "Powder Blue")],
    "Vibrant Colors": [("#FF69B4", "Hot Pink"), ("#FF1493", "Deep Pink"), ("#FF00FF", "Magenta"), ("#DC143C", "Crimson"), ("#FF6347", "Tomato")],
    "Bright Colors": [("#FFB6C1", "Pastel Pink"), ("#FFFF00", "Yellow"), ("#F0E68C", "Khaki"), ("#98FB98", "Pale Green"), ("#00FF00", "Lime Green")],
    "Muted Colors": [("#D3D3D3", "Light Grey"), ("#A9A9A9", "Dark Grey"), ("#808080", "Grey"), ("#C0C0C0", "Silver"), ("#B0C4DE", "Light Steel Blue")],
    "Bold Colors": [("#800000", "Maroon"), ("#8B0000", "Dark Red"), ("#DC143C", "Crimson"), ("#B22222", "Firebrick"), ("#FF0000", "Red")],
    "Deep Colors": [("#4B0082", "Indigo"), ("#000080", "Navy"), ("#003366", "Midnight Blue"), ("#2F4F4F", "Dark Slate Gray"), ("#800080", "Purple")]
}

@app.route('/')
def home():
    return render_template('index.html')

import base64
from io import BytesIO
from flask import render_template, request
from PIL import Image
import numpy as np
import cv2
import xgboost as xgb

@app.route('/main2', methods=['GET', 'POST'])
def main2():
    if request.method == 'POST':
        # Handle captured image (base64) or uploaded file
        if 'captured-image' in request.form and request.form['captured-image']:
            image_data = request.form['captured-image']
            image_data = image_data.split(",")[1]  # Remove the base64 header
            image = Image.open(BytesIO(base64.b64decode(image_data)))
        elif 'image' in request.files and request.files['image']:
            file = request.files['image']
            image = Image.open(file.stream)
        else:
            return render_template('main2.html', error="No image provided. Please upload or capture an image.")

        # Convert image to numpy array
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect faces using your existing face detection function
        faces = detect_faces(image_np)
        if len(faces) == 0:
            return render_template('main2.html', error="No face detected in the image. Please upload a clear image.")

        # Analyze skin tone
        x, y, w, h = faces[0]  # Use the first detected face
        skin_tone = analyze_skin_tone(image_np, x, y, w, h)

        # Ensure skin_tone is in the correct format
        skin_tone_features = [skin_tone]

        # Predict the palette index
        dmatrix_input = xgb.DMatrix(skin_tone_features)
        predicted_index = model.predict(dmatrix_input)[0]

        # Map the predicted index to a color palette
        palette_names = list(color_palettes.keys())
        palette_name = palette_names[int(predicted_index)]
        recommendations = color_palettes.get(palette_name, [])

        # Render the template with results
        return render_template('main2.html', skin_tone=skin_tone, recommendations=recommendations, palette_name=palette_name)

    return render_template('main2.html')


def detect_faces(image):
    """Detect faces using MTCNN."""
    faces = detector.detect_faces(image)
    face_boxes = []
    for face in faces:
        # Extract the bounding box coordinates for each detected face
        x, y, w, h = face['box']
        face_boxes.append((x, y, w, h))
    return face_boxes

def analyze_skin_tone(image, x, y, w, h):
    """Analyze the skin tone from the image using the face bounding box."""
    face_roi = image[y:y+h, x:x+w]
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    reshaped = face_roi.reshape((-1, 3))
    reshaped = np.float32(reshaped)

    # Convert the skin region to a more effective color space
    # Optionally use Lab or other color models for better skin tone detection
    
    # Use K-Means clustering to find dominant color
    num_clusters = 3
    _, _, centers = cv2.kmeans(
        reshaped, num_clusters, None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS
    )
    dominant_color = centers[0]  # Choose the most frequent cluster
    return tuple(map(int, dominant_color))




# Function to detect face and extract skin region
def detect_face_and_skin(image_path):
    image = cv2.imread(image_path)
    detector = MTCNN()
    results = detector.detect_faces(image)

    if len(results) == 0:
        return None, "No face detected. Please upload a clear image."

    # Get bounding box
    face = results[0]
    x, y, width, height = face['box']
    cropped_face = image[y:y + height, x:x + width]

    return cropped_face, None

# Function to detect redness
def detect_redness(skin_region):
    hsv = cv2.cvtColor(skin_region, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    redness_score = cv2.countNonZero(mask) / (skin_region.shape[0] * skin_region.shape[1])
    return redness_score

# Function to detect dryness
def detect_dryness(skin_region):
    gray = cv2.cvtColor(skin_region, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    dryness_score = np.mean(enhanced)
    return dryness_score

# Function to detect pimples
def detect_pimples(skin_region):
    hsv = cv2.cvtColor(skin_region, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours to identify pimples
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pimple_count = sum(50 < cv2.contourArea(c) < 500 for c in contours)  # Count valid pimples

    return pimple_count

@app.route('/skincare', methods=['GET', 'POST'])
def skincare():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400

        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Detect face and skin region
        skin_region, error = detect_face_and_skin(file_path)
        if error:
            return render_template('upload.html', error=error)

        # Analyze skin for redness, dryness, and pimples
        redness_score = detect_redness(skin_region)
        dryness_score = detect_dryness(skin_region)
        pimple_count = detect_pimples(skin_region)

        # Interpret analysis results
        redness_level = "High" if redness_score > 0.2 else "Moderate" if redness_score > 0.1 else "Low"
        dryness_level = "Severe" if dryness_score > 150 else "Moderate" if dryness_score > 100 else "Normal"
        pimple_level = f"{pimple_count} pimple(s)" if pimple_count > 0 else "No pimples detected"

        # Generate recommendations
        recommendations = {
            "redness": skincare_suggestions.get("redness", {}).get("severe" if redness_score > 0.2 else "mild", {}),
            "dryness": skincare_suggestions.get("dryness", {}).get("severe" if dryness_score > 150 else "mild", {}),
            "pimples": skincare_suggestions.get("pimples", {}).get("moderate" if pimple_count > 2 else "mild", {})
        }

        return render_template('results.html',
                               redness_level=redness_level,
                               dryness_level=dryness_level,
                               pimple_level=pimple_level,
                               recommendations=recommendations,
                               image_url=file_path)

    return render_template('upload.html')




# Example chatbot responses
chatbot_responses = {
    "upload": [
        "To upload a photo, click on 'Choose File' and select an image from your device. Ensure that your face is clear and well-lit for accurate analysis.",
        "Upload a front-facing image with minimal makeup and no filters for the best results. Natural light works best!"
    ],
    "redness": [
        "If you're experiencing redness, opt for green-tinted primers or foundations as they neutralize red tones on the skin. For clothing, try calming shades like soft blues or lavender to complement your look.",
        "For skin redness, avoid using products with alcohol or strong fragrances. Stick to hypoallergenic brands and wear light fabrics in soothing tones like pastels or earthy neutrals to enhance your overall appearance."
    ],
    "dryness": [
        "Dealing with dryness? Consider hydrating face masks and moisturizers containing hyaluronic acid. For clothing, choose soft and breathable fabrics like cotton or silk to avoid skin irritation.",
        "Dry skin needs extra care. Use a heavy moisturizer, especially during colder months. Opt for warm, solid colors like deep reds or mustard yellows in your wardrobe to bring warmth to your look."
    ],
    "face_shape": {
        "round": [
            "For round face shapes, try wearing V-neck tops or long necklaces to elongate the appearance of your face. Hairstyles with volume at the crown can also work wonders!",
            "Clothing with vertical patterns or fitted jackets can create a more structured look for round face shapes. Avoid rounded collars and opt for sharp, angular designs instead."
        ],
        "oval": [
            "Oval face shapes are versatile! Experiment with a variety of necklines and hairstyles. Bold earrings or statement jewelry will highlight your balanced proportions.",
            "For oval faces, almost every outfit works. Consider trying turtlenecks or high collars to draw attention to your elongated neck and symmetrical features."
        ],
        "square": [
            "For square face shapes, soft, flowy fabrics and rounded necklines work well to balance the angles of your face. Side-swept hairstyles can soften the jawline too.",
            "Add a touch of elegance with rounded jewelry and dresses with asymmetrical cuts. Avoid harsh geometric patterns to create a harmonious appearance."
        ]
    },
    "skin_tone": {
        "warm": [
            "Warm undertones shine in earthy colors like olive, mustard yellow, and terracotta. Gold jewelry complements your skin beautifully.",
            "For warm undertones, avoid cool shades like icy blues and go for rich colors like burnt orange, deep green, or coral pink for a radiant look."
        ],
        "cool": [
            "Cool undertones look stunning in jewel tones like sapphire, emerald, and royal blue. Silver accessories are a great match for your complexion.",
            "Avoid warm hues like orange or yellow. Instead, opt for pastel shades or cool grays to create a chic, polished look."
        ],
        "neutral": [
            "Neutral undertones can wear almost any color! Experiment with a mix of warm and cool shades, like lavender, taupe, or blush pink.",
            "Avoid overly bright or neon colors, and stick to muted tones or monochrome outfits for a sophisticated style."
        ]
    },
    "general_skin_care": [
        "For healthy skin, remember to cleanse, tone, and moisturize daily. Don’t forget sunscreen, even on cloudy days!",
        "Stay hydrated and eat a balanced diet rich in vitamins A and E to keep your skin glowing.",
        "Always patch-test new skincare products before applying them to your entire face to avoid adverse reactions."
    ],
    "default": [
        "I'm here to assist you with your skin analysis and style recommendations. Try asking about 'redness', 'dryness', 'face shape', or 'skin tone'!",
        "Welcome! You can ask me about dressing tips based on your skin tone or face shape, skincare suggestions, or how to upload a photo for analysis."
    ]
}

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_message = data.get("message", "").lower()

    # Simple logic to match keywords in user message
    if "upload" in user_message:
        response = choice(chatbot_responses["upload"])
    elif "redness" in user_message:
        response = choice(chatbot_responses["redness"])
    elif "dryness" in user_message:
        response = choice(chatbot_responses["dryness"])
    else:
        response = choice(chatbot_responses["default"])

    return jsonify({"reply": response})


if __name__ == "__main__":
    app.run(debug=True)