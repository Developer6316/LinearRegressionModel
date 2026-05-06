import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('salesdata.csv')
X = df[['tv','radio','newspaper']]
y = df['sales']

model = LinearRegression().fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to model.pkl")
