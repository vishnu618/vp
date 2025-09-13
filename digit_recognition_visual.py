# digit_recognition_visual.py

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load handwritten digits dataset (0–9)
digits = load_digits()
X, y = digits.data, digits.target

# Split into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple classifier
model = LogisticRegression(max_iter=10000)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

# Pick one sample to visualize
sample_index = 0
sample_image = X_test[sample_index].reshape(8, 8)

plt.imshow(sample_image, cmap="gray")
plt.title(f"True: {y_test[sample_index]} | Predicted: {model.predict([X_test[sample_index]])[0]}")
plt.axis("off")
plt.show()
