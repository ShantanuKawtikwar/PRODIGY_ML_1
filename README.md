# Housing Price Prediction Model 🏡

This repository contains a clean, production-ready, and fully documented **Linear Regression model**. The model is designed to reliably predict house prices based on three core real estate features:
1. **Square Footage** (Total above-ground living area)
2. **Number of Bedrooms** (Above grade)
3. **Number of Bathrooms** (Full baths + 0.5 * Half baths)

It has been tailored to parse and learn directly from the standard [Kaggle House Prices: Advanced Regression Techniques Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

## 📁 Project Structure

- `house_price_model.py`: The core object-oriented Python module. Includes comprehensive data loading, ETL preprocessing, model training, persistence (saving/loading), evaluation plotting, and inference logic.
- `requirements.txt`: Clean Python package dependencies necessary for execution.

*To be generated upon first execution:*
- `house_price_model.pkl`: The serialized, production-ready trained model.
- `prediction_results_plot.png`: A high-DPI resolution evaluation validation plot.

---

## 🚀 Setup & Execution 

### 1. Prerequisites
Ensure you have Python 3.7+ installed. We highly recommend using a virtual environment (`venv`).
Install the required libraries with:
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
Since the Kaggle dataset is strictly gated by authentication, you must provide it manually:
1. Visit the [Kaggle Competition Data Page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
2. Login to Kaggle and download the `train.csv` file from the data tab.
3. Place `train.csv` directly into the root of this project directory (alongside `house_price_model.py`).

### 3. Run the Model Pipeline
Execute the main script from your terminal:
```bash
python house_price_model.py
```

---

## ⚙️ What Happens Under The Hood?

When you run `house_price_model.py`, the system autonomously performs the following rigorous pipeline:
1. **ETL Process Validation**: The script automatically verifies the existence of `train.csv`. It actively maps Kaggle's technical columns (`GrLivArea`, `BedroomAbvGr`, `FullBath`, `HalfBath`) into our clean requested features dropping nulls and verifying data integrity.
2. **Training & Validation**: The data is rigorously split (80% training / 20% test) for cross-validation to prevent overfitting. A powerful Sci-kit Learn Multiple Linear Regression model is trained. Standard evaluation metrics (RMSE, MAE, R-Squared) are cleanly logged to your terminal.
3. **Formal Artifact Generation**: A detailed visualization scatter plot (`prediction_results_plot.png`) accurately assessing model variance is saved to the working directory. The trained model state is seamlessly serialized to disk as `house_price_model.pkl`.
4. **Live Inference Demonstration**: The module actively proves its capability by immediately running a dynamic prediction inference for a 4-bedroom, 2.5-bathroom, 2000 SQFT house. 

## ⚖️ Scalability & Further Integration
The robust logic is entirely encapsulated within the `HousePricePredictor` class. Once initially trained from the CSV, the instance will automatically load the local `pkl` model. 
This means you can easily implement the model into web APIs (e.g., Flask, FastAPI) and utilize the highly optimized `.predict(sqft, beds, baths)` interface for instant millisecond inference on new company property listings!
