import streamlit as st
import json
import numpy as np
import tensorflow as tf
import gdown
import zipfile

# Cache the model loading
@st.cache_resource
def load_model():
    # Saved model link
    url = "https://drive.google.com/uc?id=1m9YVs0cBRT3-j98rn7d_0DT7jwB_EXPu"
    output = 'model.zip'
    gdown.download(url, output, quiet=False)
    
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    model = tf.keras.models.load_model('best_model')
    return model

# Cache the JSON
@st.cache_data
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

user_db = load_json("user_db.json")
item_db = load_json("item_db.json")

# Function to predict rating
def predict_rating(reviewerID, itemID, model):
    item_attributes = item_db.get(itemID, {})
    user_attributes = user_db.get(reviewerID, {})
    
    category = item_attributes.get('category', 4)  # Assuming default values
    price = item_attributes.get('price', 13.71)
    userAvgRating = user_attributes.get('userAvgRating', 4)
    itemAvgRating = item_attributes.get('itemAvgRating', 4)
    review_time = user_attributes.get('unixReviewTime', 1285579290)

    reviewText_placeholder = ""
    summary_placeholder = ""
    
    prediction_inputs = {
        'reviewer_id': np.array([reviewerID], dtype=np.int32),
        'item_id': np.array([itemID], dtype=np.int32),
        'category': np.array([category], dtype=np.int32),
        'price': np.array([price], dtype=np.float32),
        'paid_price': np.array([price], dtype=np.float32),  # Assuming you want to reuse price here
        "unixReviewTime": np.array([review_time], dtype=np.float32),  
        'userAvgRating': np.array([userAvgRating], dtype=np.float32),
        'itemAvgRating': np.array([itemAvgRating], dtype=np.float32),
        'review_text': np.array([reviewText_placeholder]),
        'summary': np.array([summary_placeholder]),
    }
    
    prediction = model.predict(prediction_inputs)
    return prediction.item()

model = load_model()

# App interface
st.title('Music Rating Prediction - Amazon Review')

# Example input values
example_reviewerID = "61658"  # Example reviewerID
example_itemID = "5000"  # Example itemID

# Inputs
reviewerID = st.text_input('Reviewer ID', value=example_reviewerID)
itemID = st.text_input('Item ID', value=example_itemID)

# Button
if st.button('Predict Rating'):
    prediction = predict_rating(reviewerID, itemID, model)
    st.write(f'Predicted Rating: {prediction:.2f} ⭐')