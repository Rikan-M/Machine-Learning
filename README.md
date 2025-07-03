# 📦 BatchRegressor

**BatchRegressor** is a lightweight, custom-built linear regression model implemented from scratch using only NumPy. It supports training using **mini-batch gradient descent**, making it a great educational tool for understanding the inner workings of machine learning models without relying on external ML libraries.

---

## 🚀 Features

- ✅ Linear regression with batch gradient descent
- ✅ Mean Squared Error (MSE) loss function
- ✅ Custom learning rate support
- ✅ Weight and bias initialization from scratch
- ✅ Modular and clean class structure
- ✅ No machine learning libraries required (pure NumPy)
- ✅ Easy to extend or integrate

---

## 🧠 Why This Project?

This project is built for **learning purposes** — it helps you understand:

- How gradient descent works
- How loss functions and gradients are calculated
- How weights and bias are updated over time
- The concept of batching in machine learning

It’s ideal for:

- Students
- Beginners in ML
- Interviews / Demos
- Python + NumPy practice

---

## 📁 File Structure

├── BatchRegressor.py # The main BatchRegressor class
└── README.md # Project documentation

---

## 📦 Installation

No installation required. Just make sure you have Python and NumPy installed:

```bash
pip install numpy
🛠️ How to Use
from BatchRegressor import BatchRegressor
import numpy as np
import math

# Sample Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([[2], [4], [6], [8], [10]])

# Initialize model
model = BatchRegressor(learning_rate=0.01)

# Train the model
model.fit(X, y, batch_size=2, epochs=100)

# Make predictions
predictions = model.predict(X)
print(predictions)

🧪 Example Output

Epoch1 : ------
Error :  10.234
...
Epoch100 : ------
Error :  0.005

[ 2.01  4.01  6.01  8.00  9.98]
📌 Future Improvements
Add support for other loss functions (e.g., MAE)

Add L2 regularization

Add visualizations for loss per epoch

Save and load model weights

🤝 Contributing
Feel free to fork, improve, or suggest features via pull requests!

📜 License
This project is open-source and free to use for educational or commercial purposes. License: MIT

👨‍💻 Author
Rikan Maji
Python & ML Enthusiast
```
