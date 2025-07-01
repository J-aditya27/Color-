import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import cv2
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# Original training data
X_train = np.array([
    [240, 200, 180],  # Light skin tone (fair)
    [220, 180, 150],  # Light skin tone (fair)
    [150, 120, 100],  # Medium skin tone
    [180, 130, 100],  # Medium skin tone
    [100, 70, 60],    # Dark skin tone
    [70, 50, 40],     # Dark skin tone
    [250, 230, 220],  # Very fair skin tone
    [180, 150, 130],  # Olive skin tone
    [120, 80, 60],    # Tan skin tone
    [60, 40, 30],     # Very dark skin tone
    [220, 185, 165],  # Fair skin with warm undertones
    [200, 160, 120],  # Medium skin with warm undertones
    [90, 60, 50],     # Dark skin with neutral undertones
    [210, 180, 160],  # Olive skin with neutral undertones
    [130, 100, 90],   # Tan skin with warm undertones
    [50, 30, 20],     # Very dark skin with cool undertones
    [240, 215, 180],  # Fair skin with cool undertones
    [160, 120, 100],  # Medium skin with cool undertones
    [110, 90, 70],    # Dark skin with warm undertones
    [180, 140, 120],  # Olive skin with warm undertones
    [240, 220, 200],  # Fair skin with vibrant undertones
    [180, 100, 80],   # Dark skin with bright tones
    [200, 180, 160],  # Light skin with muted tones
    [120, 100, 80],   # Medium skin with deep tones
    [170, 150, 130],  # Olive skin with bold colors
    [130, 110, 90],   # Tan skin with light colors
    [80, 60, 50],     # Very dark skin with dark tones
    [210, 190, 170],  # Fair skin with deep tones
    [180, 160, 140],  # Medium skin with bold colors
    [100, 80, 60],    # Dark skin with bright colors
    [200, 190, 170],  # Olive skin with vibrant tones
])

y_train = np.array([
    "Pastels",         # Suitable for light/fair skin
    "Pastels",         # Suitable for light/fair skin
    "Earthy Tones",    # Suitable for medium skin
    "Earthy Tones",    # Suitable for medium skin
    "Warm Tones",      # Suitable for dark skin
    "Warm Tones",      # Suitable for dark skin
    "Neutrals",        # Suitable for very fair skin
    "Earthy Tones",    # Suitable for olive skin
    "Pastels",         # Suitable for tan skin
    "Warm Tones",      # Suitable for very dark skin
    "Pastels",         # Suitable for fair skin with warm undertones
    "Earthy Tones",    # Suitable for medium skin with warm undertones
    "Warm Tones",      # Suitable for dark skin with neutral undertones
    "Neutrals",        # Suitable for olive skin with neutral undertones
    "Pastels",         # Suitable for tan skin with warm undertones
    "Cool Tones",      # Suitable for very dark skin with cool undertones
    "Pastels",         # Suitable for fair skin with cool undertones
    "Earthy Tones",    # Suitable for medium skin with cool undertones
    "Warm Tones",      # Suitable for dark skin with warm undertones
    "Earthy Tones",    # Suitable for olive skin with warm undertones
    "Vibrant Colors",  # Suitable for fair skin with vibrant undertones
    "Bright Colors",   # Suitable for dark skin with bright tones
    "Muted Colors",    # Suitable for light skin with muted tones
    "Deep Colors",     # Suitable for medium skin with deep tones
    "Bold Colors",     # Suitable for olive skin with bold colors
    "Light Colors",    # Suitable for tan skin with light colors
    "Dark Colors",     # Suitable for very dark skin with dark tones
    "Deep Colors",     # Suitable for fair skin with deep tones
    "Bold Colors",     # Suitable for medium skin with bold colors
    "Bright Colors",   # Suitable for dark skin with bright colors
    "Vibrant Colors",  # Suitable for olive skin with vibrant tones
])

# Function to add noise and generate more unique data
def generate_more_data(X, y, num_new_samples=1000):
    new_X = []
    new_y = []
    for i in range(num_new_samples):
        # Randomly pick a skin tone from the original data
        idx = np.random.randint(len(X))
        base_X = X[idx]
        base_y = y[idx]
        
        # Add random noise to RGB values to generate a new sample
        noise = np.random.randint(-10, 10, size=(3,))
        new_X.append(base_X + noise)
        new_y.append(base_y)
        
    return np.array(new_X), np.array(new_y)

# Generate 1000 new samples
X_train_expanded, y_train_expanded = generate_more_data(X_train, y_train, num_new_samples=1000)

# Combine the original and new data
X_train_combined = np.vstack((X_train, X_train_expanded))
y_train_combined = np.hstack((y_train, y_train_expanded))

# Label Encoding for categorical labels (string to numeric conversion)
label_encoder = LabelEncoder()
y_train_combined_encoded = label_encoder.fit_transform(y_train_combined)

# Function to convert RGB to LAB color space
def rgb_to_lab(rgb):
    rgb = np.uint8([[rgb]])  # Convert to 2D array
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab[0][0]  # Return Lab values

# Apply RGB to Lab conversion to the dataset
X_train_lab = np.array([rgb_to_lab(rgb) for rgb in X_train_combined])

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_train_lab, y_train_combined_encoded, test_size=0.2, random_state=42)

# Convert to DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters for XGBoost
params = {
    'objective': 'multi:softmax',
    'num_class': len(np.unique(y_train)),  # Number of classes in your labels
    'max_depth': 6,
    'learning_rate': 0.1,
    'eval_metric': 'merror'
}

# Train the model
model = xgb.train(params, dtrain, num_boost_round=100)

# Save the trained model using joblib
joblib.dump(model, 'ml/recommendation_model.joblib')

print("Model trained and saved successfully!")
