from flask import Flask, request, render_template, redirect, url_for
import os
from main import analyze_tennis

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    video_url = None
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Perform analysis and get the path to the generated video
        analyze_tennis(file_path, 'static/output.mp4')
        # Convert the path to a URL for static files
        video_url = url_for('static', filename='output.mp4')

    return render_template('index.html', video_url=video_url)

if __name__ == '__main__':
    app.run(debug=True)