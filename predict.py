import pickle
import sys

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

tv, radio, newspaper = map(float, sys.argv[1:4]) if len(sys.argv) > 3 else (100, 100, 100)
pred = model.predict([[tv, radio, newspaper]])[0]
print(f"Predicted sales: {pred:.2f}")