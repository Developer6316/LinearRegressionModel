# 📊 Linear Regression – Advertising Sales Prediction

A professional machine learning project that predicts sales based on TV, radio, and newspaper advertising spend using linear regression. Achieves **R² = 0.81** on a realistic synthetic dataset (205 rows).

---

## 🎯 Overview

This repository demonstrates a complete ML workflow:
- **Data Loading & Preprocessing**: Clean and prepare advertising data
- **Model Training**: Build and train a linear regression model
- **Evaluation**: Comprehensive performance metrics (R², MSE)
- **Prediction**: Make predictions on new data points
- **Model Persistence**: Save and load trained models

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | 0.8102 |
| **Intercept** | 3.459 |
| **TV Coefficient** | 0.0445 |
| **Radio Coefficient** | 0.1747 |
| **Newspaper Coefficient** | 0.0174 |

### Model Equation
```
sales = 3.459 + 0.0445×tv + 0.1747×radio + 0.0174×newspaper
```

### Example Prediction
```
Input:  tv=100, radio=100, newspaper=100
Output: Predicted sales = 27.12
```

---

## 📁 Project Structure

```
LinearRegressionModel/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── data/
│   └── salesdata.csv        # Training dataset (205 rows)
├── src/
│   ├── __init__.py
│   ├── train.py            # Model training script
│   ├── predict.py          # Prediction script
│   └── models.py           # Core ML functions (refactored)
├── models/
│   └── model.pkl           # Trained model (generated after training)
├── notebooks/
│   └── model1.ipynb        # Jupyter notebook for exploration
└── LICENSE                 # MIT License
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### 1. Clone the Repository
```bash
git clone https://github.com/Developer6316/LinearRegressionModel.git
cd LinearRegressionModel
```

### 2. Create Virtual Environment (Recommended)

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python src/train.py
```
**Output**: Generates `models/model.pkl`

### 5. Make Predictions
```bash
# Using default values (100, 100, 100)
python src/predict.py

# Custom values
python src/predict.py 150 75 50
```

---

## 📊 Dataset Information

**File**: `data/salesdata.csv`
- **Rows**: 205 samples
- **Columns**: `tv`, `radio`, `newspaper`, `sales`
- **Data Generation**: Synthetic data based on real-world advertising patterns
  ```
  sales = 3.5 + 0.045×tv + 0.18×radio + 0.015×newspaper + noise
  ```

---

## 💻 Code Files

### `src/train.py`
Trains the linear regression model on the advertising data.
```python
python src/train.py
```
**Outputs**: `models/model.pkl`

### `src/predict.py`
Loads the trained model and makes predictions.
```python
python src/predict.py [tv] [radio] [newspaper]
```
**Example**: `python src/predict.py 100 100 100`

### `src/models.py` (Refactored)
Core machine learning functions with proper error handling and logging.

### `notebooks/model1.ipynb`
Jupyter notebook for data exploration and model analysis.

---

## 🔍 Model Details

### Algorithm
- **Method**: Ordinary Least Squares (OLS) Linear Regression
- **Library**: scikit-learn
- **Train/Test Split**: 90/10
- **Test Set Size**: ~21 samples

### Feature Importance
1. **Radio** (0.1747) - Most influential advertising channel
2. **TV** (0.0445) - Moderate impact
3. **Newspaper** (0.0174) - Smallest impact

---

## ⚠️ Limitations & Future Improvements

### Current Limitations
- Linear model assumes linear relationships between features and sales
- Limited to 3 advertising channels
- Small dataset (205 samples)
- No feature scaling or normalization

### Potential Enhancements
- [ ] Add polynomial features for non-linear relationships
- [ ] Implement cross-validation
- [ ] Add hyperparameter tuning
- [ ] Support for multiple regression models (Ridge, Lasso, etc.)
- [ ] Web API for predictions (Flask/FastAPI)
- [ ] Unit tests and integration tests
- [ ] Data visualization and analysis
- [ ] Model explainability (SHAP values)

---

## 🧪 Testing

```bash
# Example test prediction
python src/predict.py 100 100 100
# Expected output: Predicted sales: 27.12
```

---

## 📝 License

MIT License - See `LICENSE` file for details

---

## 👨‍💻 Author

**Developer6316**
- GitHub: [@Developer6316](https://github.com/Developer6316)

---

## 📧 Feedback & Support

Have suggestions or found issues? Please open a [GitHub Issue](https://github.com/Developer6316/LinearRegressionModel/issues)

---

## 🔗 Resources

- [scikit-learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [Git & GitHub Basics](https://docs.github.com/en/get-started)
