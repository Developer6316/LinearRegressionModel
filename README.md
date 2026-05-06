# Linear Regression – Advertising Sales Prediction

Predict sales based on TV, radio, and newspaper advertising spend using a simple linear regression model.  
Achieves **R² = 0.81** on a realistic synthetic dataset (205 rows).

---

## 📊 Model Performance

| Metric       | Value |
|--------------|-------|
| R² Score     | 0.8102 |
| Intercept    | 3.459 |
| TV coefficient | 0.0445 |
| Radio coefficient | 0.1747 |
| Newspaper coefficient | 0.0174 |

**Formula:**  
`sales = 3.459 + 0.0445×tv + 0.1747×radio + 0.0174×newspaper`

**Example:** `tv=100, radio=100, newspaper=100` → **Predicted sales = 27.12**

---

## 📁 Dataset

`tet.csv` contains 205 rows with columns: `tv`, `radio`, `newspaper`, `sales`.  
Generated from:  
`sales = 3.5 + 0.045×tv + 0.18×radio + 0.015×newspaper + noise`

---

## 🔧 Setup (works on Windows, macOS, Linux)

### 1. Clone the repository
```bash
git clone https://github.com/Developer6316/your-repo-name.git
cd your-repo-name.
```
### 2. Create a virtual environment(Recommended)
 #### Windows
 ```bash
python -m venv venv
venv\Scripts\activate
```
 #### Linux/Mac OS
 ```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install numpy pandas scikit-learn
```
#### In some cases in Linux(i.e Ubuntu,Kali,etc)
```bash
sudo apt install python3-numpy python3-pandas python3-sklearn
```
### 4. Train the model (generates model.pkl)
 ```bash
python train.py
```
### 5. Make Predictions
```bash
python predict.py 100 100 100
```
