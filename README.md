# 🧠 Customer Segmentation with K-Means + FastAPI

This project performs customer segmentation using the K-Means clustering algorithm. It identifies customer groups based on behavioral and demographic patterns, and serves predictions through a FastAPI backend.

---

## 📊 Features

- Exploratory Data Analysis (EDA) on customer dataset
- Feature engineering and preprocessing
- K-Means clustering to group customers
- Visualization of clusters
- FastAPI app for real-time cluster prediction
- Optional Swagger UI for API interaction

---

## 📁 Project Structure
```bash
├── data/ # Raw and processed datasets
├── eda/ # EDA notebooks or scripts
├── model/ # Trained model artifacts (e.g., scaler, kmeans.pkl)
├── api/ # FastAPI app (main.py, routes, utils)
├── requirements.txt # Python dependencies
├── README.md # Project overview
```


---

## 🚀 Getting Started

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/customer-segmentation-api.git
2. **install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run FastAPI Server**
   ```bash
   uvicorn api.main:app --reload

5. **Test API**
   ```bash
   http://127.0.0.1:8000/docs

#Example of API input
{
  "age": 29,
  "annual_income": 50000,
  "spending_score": 65
}

 Output
-The cluster label that this customer belongs to, e.g., "cluster": 2.






# Data Folder

The dataset used in this project can be downloaded from:

[UCI Machine Learning Repository – Online Retail II Data Set](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)

⚠️ Note: This folder is intentionally left empty and is excluded from version control via `.gitignore`.
