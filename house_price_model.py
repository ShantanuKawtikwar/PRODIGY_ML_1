import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import joblib

# Set up logging for reporting transparency
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HousePricePredictor:
    """
    A Production-Ready Linear Regression Model designed to predict house prices 
    using square footage (GrLivArea), bedrooms (BedroomAbvGr), and bathrooms
    (FullBath + HalfBath). Based on the Kaggle House Prices Advanced Regression 
    Techniques dataset.
    """
    
    def __init__(self, model_path='house_price_model.pkl'):
        """
        Initializes the HousePricePredictor.
        
        Args:
            model_path (str): The file path where the trained model will be saved/loaded.
        """
        self.model = LinearRegression()
        self.features = ['SquareFootage', 'Bedrooms', 'Bathrooms']
        self.target = 'SalePrice'
        self.model_path = model_path
        self.is_trained = False

    def load_and_preprocess_data(self, data_path):
        """
        Loads the Kaggle dataset and extracts/engineers the requested features.
        
        Args:
            data_path (str): Path to the CSV dataset (e.g., 'train.csv').
            
        Returns:
            pd.DataFrame: A preprocessed dataframe ready for training.
        """
        logger.info(f"Loading data from {data_path}...")
        try:
            df = pd.read_csv(data_path)
            
            # Check if required raw columns exist
            required_raw_cols = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'SalePrice']
            missing_cols = [col for col in required_raw_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in dataset: {missing_cols}")
                
            # Feature Engineering mapping Kaggle columns to requested features
            # Square footage -> GrLivArea (Above grade living area square feet)
            # Bedrooms -> BedroomAbvGr
            # Bathrooms -> FullBath + 0.5 * HalfBath (Standard real estate calculation)
            df['SquareFootage'] = df['GrLivArea']
            df['Bedrooms'] = df['BedroomAbvGr']
            df['Bathrooms'] = df['FullBath'] + (0.5 * df['HalfBath'])
            
            # Select final subset and drop NaNs
            processed_df = df[self.features + [self.target]].dropna()
            logger.info(f"Data perfectly loaded and preprocessed. Valid samples: {processed_df.shape[0]}")
            return processed_df
            
        except FileNotFoundError:
            logger.error(f"Could not find {data_path}. Please ensure the Kaggle dataset is downloaded.")
            raise
        except Exception as e:
            logger.error(f"Error loading and preprocessing data: {str(e)}")
            raise

    def train_model(self, data, test_size=0.2, random_state=42):
        """
        Splits data, trains the linear regression model, and evaluates it.
        
        Args:
            data (pd.DataFrame): The preprocessed dataset.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data.
            
        Returns:
            dict: Evaluation metrics (RMSE, MAE, R2-Score)
        """
        logger.info("Splitting data into training and testing sets (80/20)...")
        X = data[self.features]
        y = data[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        logger.info("Training the Linear Regression model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        logger.info("Evaluating model performance on test set...")
        predictions = self.model.predict(X_test)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
            'MAE': mean_absolute_error(y_test, predictions),
            'R2_Score': r2_score(y_test, predictions)
        }
        
        logger.info(f"Model Evaluation Results: RMSE: ${metrics['RMSE']:,.2f} | MAE: ${metrics['MAE']:,.2f} | R2 Score: {metrics['R2_Score']:.4f}")
        
        # Evaluation Visualization
        self._plot_actual_vs_predicted(y_test, predictions)
        
        return metrics

    def predict(self, square_footage, bedrooms, bathrooms):
        """
        Predicts the house price given specific features.
        
        Args:
            square_footage (float/int): The square footage of the house.
            bedrooms (int): Number of bedrooms.
            bathrooms (float/int): Number of bathrooms.
            
        Returns:
            float: Predicted house price.
        """
        if not self.is_trained:
            logger.warning("Model is not trained in current memory. Attempting to load from disk...")
            self.load_model()
            
        input_data = pd.DataFrame([{
            'SquareFootage': square_footage,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms
        }])
        
        prediction = self.model.predict(input_data)[0]
        return round(prediction, 2)

    def save_model(self):
        """Saves the trained model to disk for later inference."""
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model successfully saved to '{self.model_path}'.")

    def load_model(self):
        """Loads a formally trained model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            logger.info(f"Model successfully loaded from '{self.model_path}'.")
        else:
            raise FileNotFoundError(f"No trained model found at '{self.model_path}'. Train it first.")

    def _plot_actual_vs_predicted(self, actual, predicted):
        """Creates a scatter plot comparing actual versus predicted values as a tangible artifact."""
        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=actual, y=predicted, alpha=0.6, color='blue', edgecolor=None)
            
            # Perfect prediction diagonal line
            plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2, label='Perfect Prediction')
            
            plt.title('Linear Regression: Actual vs Predicted House Prices', fontsize=14)
            plt.xlabel('Actual Sale Price ($)', fontsize=12)
            plt.ylabel('Predicted Sale Price ($)', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plot_path = 'prediction_results_plot.png'
            plt.savefig(plot_path, dpi=300)
            logger.info(f"Saved formal evaluation plot to '{plot_path}'.")
        except Exception as e:
            logger.warning(f"Could not save evaluation plot: {str(e)}")

if __name__ == "__main__":
    print(f"{'='*50}")
    print(f"{' House Price Prediction Model Execution ':^50}")
    print(f"{'='*50}")
    
    # 1. Initialize Predictor
    predictor = HousePricePredictor()
    
    data_path = 'train.csv'
    
    if os.path.exists(data_path):
        # 2. Load and Preprocess Data
        df = predictor.load_and_preprocess_data(data_path)
        
        # 3. Train the Model
        print(f"\n[{pd.Timestamp.now().strftime('%H:%M:%S')}] Commencing model training...")
        metrics = predictor.train_model(df)
        
        # 4. Demonstrate a Live Inference Prediction
        print("\n--- COMPANY SAMPLE INFERENCE TEST ---")
        sample_sqft = 2000
        sample_beds = 4
        sample_baths = 2.5
        predicted_price = predictor.predict(sample_sqft, sample_beds, sample_baths)
        
        print(f"Input Features: {sample_sqft} sqft | {sample_beds} beds | {sample_baths} baths")
        print(f"System Predicted Price: ${predicted_price:,.2f}")
        print(f"-------------------------------------\n")
        print("Execution complete. Ready for production.")
    else:
        logger.error(f"Dataset '{data_path}' missing in root directory!")
        print("\n[ACTION REQUIRED]")
        print("Please download the Kaggle dataset ('train.csv') to this directory before running.")
        print("URL: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data")
