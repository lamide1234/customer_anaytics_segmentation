from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and scaler
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Cluster info
cluster_insights = {
    0: {
        "label": "Retain",
        "color": "Blue",
        "description": "High-value customers who purchase regularly. Focus on retention.",
        "action": "Implement loyalty programs, personalized offers, and regular engagement."
    },
    1: {
        "label": "Re-Engage",
        "color": "Orange",
        "description": "Lower-value, infrequent buyers who haven’t purchased recently.",
        "action": "Use targeted marketing, discounts, or reminders to encourage purchases."
    },
    2: {
        "label": "Nurture",
        "color": "Green",
        "description": "Least active, lowest-value but recently purchased. Likely new or hesitant.",
        "action": "Build relationships, offer incentives, and provide excellent service."
    },
    3: {
        "label": "Reward",
        "color": "Red",
        "description": "Most loyal and high-frequency buyers. Reward their loyalty.",
        "action": "Provide exclusive offers and robust loyalty programs."
    }
}

# Define input schema
class CustomerData(BaseModel):
    MonetaryValue: float
    Frequency: float
    Recency: float

# Initialize app
app = FastAPI(title="Customer Segmentation API")

@app.post("/predict")
def predict_cluster(data: CustomerData):
    # Prepare data for model
    features = np.array([[data.MonetaryValue, data.Frequency, data.Recency]])
    scaled_features = scaler.transform(features)

    # Predict cluster
    cluster = int(kmeans.predict(scaled_features)[0])
    insight = cluster_insights[cluster]

    # Return response
    return {
        "cluster": cluster,
        "label": insight["label"],
        "color": insight["color"],
        "description": insight["description"],
        "action": insight["action"]
    }
