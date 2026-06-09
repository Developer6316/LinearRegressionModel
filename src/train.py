"""Train the linear regression model on advertising sales data."""

import os
from sklearn.model_selection import train_test_split
from src.models import load_data, train_model, evaluate_model, save_model


def main():
    """Main training pipeline."""
    # Define paths
    data_path = 'data/salesdata.csv'
    model_path = 'models/model.pkl'
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load data
    print("Loading data...")
    df, X, y = load_data(data_path)
    print(f"\nData Summary:")
    print(df.describe())
    
    # Split data
    print("\nSplitting data (90/10 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"\nPerformance Metrics:")
    print(f"  R² Score: {metrics['r2_score']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    
    # Save model
    print(f"\nSaving model to {model_path}...")
    save_model(model, model_path)
    
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
