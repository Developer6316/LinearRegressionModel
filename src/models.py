"""Core machine learning functions for sales prediction model."""

import pickle
import logging
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and prepare data for model training.
    
    Args:
        filepath: Path to CSV file containing training data
        
    Returns:
        Tuple of (dataframe, features, target)
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded data from {filepath} with shape {df.shape}")
        
        # Extract features and target
        X = df[['tv', 'radio', 'newspaper']].values
        y = df['sales'].values
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return df, X, y
    except FileNotFoundError:
        logger.error(f"Data file not found: {filepath}")
        raise
    except KeyError as e:
        logger.error(f"Missing required columns: {e}")
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Train a linear regression model.
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training target values (n_samples,)
        
    Returns:
        Trained LinearRegression model
    """
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        logger.info(f"Model trained successfully")
        logger.info(f"Coefficients: tv={model.coef_[0]:.4f}, "
                   f"radio={model.coef_[1]:.4f}, "
                   f"newspaper={model.coef_[2]:.4f}")
        logger.info(f"Intercept: {model.intercept_:.4f}")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def evaluate_model(model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model performance on test data.
    
    Args:
        model: Trained LinearRegression model
        X_test: Test features
        y_test: Test target values
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        metrics = {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'predictions': y_pred,
            'actuals': y_test
        }
        
        logger.info(f"Model Evaluation - R²: {r2:.4f}, RMSE: {rmse:.4f}")
        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def save_model(model: LinearRegression, filepath: str) -> None:
    """Save trained model to disk using pickle.
    
    Args:
        model: Trained model to save
        filepath: Path where model will be saved
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def load_model(filepath: str) -> LinearRegression:
    """Load trained model from disk.
    
    Args:
        filepath: Path to saved model file
        
    Returns:
        Loaded LinearRegression model
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def make_prediction(model: LinearRegression, tv: float, radio: float, newspaper: float) -> float:
    """Make a sales prediction for given advertising spend.
    
    Args:
        model: Trained LinearRegression model
        tv: TV advertising spend
        radio: Radio advertising spend
        newspaper: Newspaper advertising spend
        
    Returns:
        Predicted sales value
    """
    try:
        prediction = model.predict([[tv, radio, newspaper]])[0]
        logger.info(f"Prediction for (tv={tv}, radio={radio}, newspaper={newspaper}): {prediction:.2f}")
        return prediction
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise
