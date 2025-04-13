from flask import Flask, request, jsonify, render_template, redirect, url_for
from models.fake_news_model import FakeNewsModel
from preprocessing.video_preprocessing import detect_deepfake_with_labels
import os

app = Flask(__name__)

# Initialize Models
news_model = FakeNewsModel()

# Upload folder for videos
UPLOAD_FOLDER = 'uploads/dataset/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- Home Route ----
@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

# ---- Form Submission for Fake News Detection ----
@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle form submission for fake news detection."""
    text = request.form.get('text', '')

    if not text:
        return render_template('index.html', error="Please enter some text.")

    # Predict fake news
    result = news_model.predict(text)

    return render_template('result.html', text=text, label=result['label'], confidence=result['confidence'])

# ---- Fake News Detection (API) ----
@app.route('/predict-fake-news', methods=['POST'])
def predict_fake_news():
    """Route for API-based fake news detection."""
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = news_model.predict(text)

    return jsonify({
        "label": result["label"],
        "confidence": result["confidence"]
    })

# ---- Video Upload Route ----
@app.route('/upload-video', methods=['POST'])
def upload_video():
    """Route for uploading video files."""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded video
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    return jsonify({"message": "Video uploaded successfully", "video_path": video_path})

# ---- Trigger Deepfake Detection ----
@app.route('/detect-deepfake', methods=['GET'])
def detect_deepfake():
    """Route to trigger deepfake detection."""
    results = detect_deepfake_with_labels()

    return jsonify({
        "message": "Deepfake detection completed",
        "results": results
    })

# ---- Run the App ----
if __name__ == '__main__':
    app.run(debug=True)


