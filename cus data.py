import pandas as pd
import numpy as np

length = 1000

def generate_numerical_data():
    return {
        "calories": np.random.randint(200, 300, size=length),
        "fats": np.random.uniform(1, 10, size=length),
        "proteins": np.random.uniform(2, 20, size=length),
        "carbs": np.random.uniform(10, 70, size=length),
        "temperature": np.random.uniform(0, 50, size=length),
        "humidity": np.random.uniform(10, 90, size=length),
        "shelf_life": np.random.randint(1, 10, size=length),
        "cost": np.random.uniform(30, 60, size=length),
        "sustainability_fact": np.random.uniform(0, 11, size=length),
        "potassium(mg)": np.random.randint(100, 200, size=length),
        "sodium(mg)": np.random.randint(400, 600, size=length),
    }

def calculate_rating(row):
    score = 0
    score -= 1 if row["calories"] > 95 else +1
    score += 1 if row["proteins"] > 2 else -1
    score -= 1 if row["fats"] > 2 else 0
    score += 1 if row["potassium(mg)"] > 30 else 0
    score += 1 if row["sodium(mg)"] < 180 else 0
    score += 1 if row["carbs"] < 18 else 0

    if row["sustainability_fact"] < 3:
        score -= 1
    elif row["sustainability_fact"] > 6:
        score += 2

    if row["cost"] < 35:
        score += 2
    elif row["cost"] < 40:
        score += 1
    elif row["cost"] < 45:
        score -= 1
    elif row["cost"] > 45:
        score -= 2

    if row["temperature"] > 180 or row["humidity"] > 60:
        score += 1

    if row["shelf_life"] > 4:
        score += 1
    else:
        score -= 1

    return max(1, min(10, 5 + score))

numerical_data = generate_numerical_data()

df = pd.DataFrame(numerical_data)

df["product"] = "bread"

df["rating"] = df.apply(calculate_rating, axis=1)

df.to_csv("food_dataset.csv", index=False)
print("Dataset generated and saved as food_dataset.csv")          