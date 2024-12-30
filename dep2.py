import streamlit as st
import numpy as np
import pickle
import random
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")

# Load the scaler and model
def load_artifacts():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("food_rating_model.pkl", "rb") as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_artifacts()

# Lightweight text generator using Markov-like chaining
def lightweight_text_generator(product_ratings):
    phrases = [
        ["Product", "{best_product}", "is highly praised for", "nutritional value."],
        ["Considering its", "shelf life", "and", "sustainability,", "Product", "{best_product}", "stands out."],
        ["Among the options,", "Product", "{best_product}", "excels", "in customer satisfaction."],
        ["For affordability and", "health benefits,", "Product", "{best_product}", "is a great choice."],
    ]
    random.shuffle(phrases)
    sentence = " ".join(random.choice(phrases))
    additional_details = (
        f"The ratings are: Product 1 - {round(product_ratings[0], 2)}, Product 2 - {round(product_ratings[1], 2)}, "
        f"Product 3 - {round(product_ratings[2], 2)}."
    )
    return additional_details + " " + sentence.format(best_product=np.argmax(product_ratings) + 1)

# App title and description
st.title("Food Rating Comparison and Recommendation")
st.write("Compare ratings of three food products and get a detailed recommendation.")

# Function to create input fields for product details
def get_product_inputs(product_id):
    with st.expander(f"Enter Product {product_id} Details"):
        calories = st.slider(f"Calories (Product {product_id})", 200, 300, 250)
        fats = st.slider(f"Fats (g) (Product {product_id})", 1.0, 10.0, 5.0)
        proteins = st.slider(f"Proteins (g) (Product {product_id})", 2.0, 20.0, 10.0)
        carbs = st.slider(f"Carbs (g) (Product {product_id})", 10.0, 70.0, 40.0)
        temperature = st.slider(f"Temperature (°C) (Product {product_id})", 0.0, 50.0, 25.0)
        humidity = st.slider(f"Humidity (%) (Product {product_id})", 10.0, 90.0, 50.0)
        shelf_life = st.slider(f"Shelf Life (days) (Product {product_id})", 1, 10, 5)
        cost = st.slider(f"Cost (₹) (Product {product_id})", 30.0, 60.0, 45.0)
        sustainability_fact = st.slider(f"Sustainability Factor (Product {product_id})", 0.0, 10.0, 5.0)
        potassium = st.slider(f"Potassium (mg) (Product {product_id})", 100, 200, 150)
        sodium = st.slider(f"Sodium (mg) (Product {product_id})", 400, 600, 500)
    return [calories, fats, proteins, carbs, temperature, humidity, shelf_life, cost, sustainability_fact, potassium, sodium]

# Input for three products
st.header("Product Inputs")
product_1 = get_product_inputs(1)
product_2 = get_product_inputs(2)
product_3 = get_product_inputs(3)

# Compare ratings
if st.button("Compare Ratings and Generate Recommendation"):
    # Prepare input data for prediction
    input_data = np.array([product_1, product_2, product_3])
    input_scaled = scaler.transform(input_data)
    ratings = model.predict(input_scaled)

    # Display ratings
    st.subheader("Product Ratings")
    for idx, rating in enumerate(ratings, start=1):
        st.write(f"Product {idx}: {round(rating, 2)}")

    # Identify the best product
    best_product = np.argmax(ratings) + 1
    st.success(f"Recommended Product: Product {best_product}")

    # Generate recommendation summary
    detailed_recommendation = lightweight_text_generator(ratings)
    st.subheader("Recommendation Summary")
    st.text_area("Detailed Recommendation", value=detailed_recommendation, height=200)

# Footer
st.markdown("---")
st.write("Powered by Streamlit and Python")
