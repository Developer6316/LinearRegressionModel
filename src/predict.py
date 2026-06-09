"""Make predictions using the trained linear regression model."""

import sys
import os
from src.models import load_model, make_prediction


def main():
    """Main prediction pipeline."""
    model_path = 'models/model.pkl'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("Please run 'python src/train.py' first to train the model.")
        sys.exit(1)
    
    # Load model
    model = load_model(model_path)
    
    # Parse command line arguments
    if len(sys.argv) >= 4:
        try:
            tv = float(sys.argv[1])
            radio = float(sys.argv[2])
            newspaper = float(sys.argv[3])
        except ValueError:
            print("❌ Error: Arguments must be numeric values")
            print("Usage: python src/predict.py [tv] [radio] [newspaper]")
            sys.exit(1)
    else:
        # Default values
        tv, radio, newspaper = 100.0, 100.0, 100.0
        print("Using default values: tv=100, radio=100, newspaper=100")
    
    # Make prediction
    print(f"\nInput Values:")
    print(f"  TV Advertising: ${tv:.2f}")
    print(f"  Radio Advertising: ${radio:.2f}")
    print(f"  Newspaper Advertising: ${newspaper:.2f}")
    
    predicted_sales = make_prediction(model, tv, radio, newspaper)
    
    print(f"\n🎯 Predicted Sales: ${predicted_sales:.2f}")


if __name__ == '__main__':
    main()
