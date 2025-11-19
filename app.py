from flask import Flask, request, jsonify
import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset and split
data = fetch_olivetti_faces()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Load trained model
model = joblib.load("savedmodel.pth")

@app.route("/")
def home():
    return "ML Model API is running!"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({"message": "Send a POST request with JSON: {\"image\": [...]}"})
    
    if request.method == "POST":
        try:
            data = request.get_json()
            img_vector = data["image"]
            prediction = model.predict([img_vector])
            return jsonify({"prediction": int(prediction[0])})
        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)