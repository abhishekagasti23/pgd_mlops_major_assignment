import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = fetch_olivetti_faces()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Load model
model = joblib.load("savedmodel.pth")

# Test accuracy
pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, pred))
