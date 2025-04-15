from flask import Flask, render_template, request
from scripts.voice_cognitive_pipeline import process_audio_file
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio_file' not in request.files:
        return "No file uploaded", 400

    file = request.files['audio_file']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = process_audio_file(filepath)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
