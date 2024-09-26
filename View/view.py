# View/view.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Controller')))
from controller import convert_images

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/convert', methods=['POST'])
def handle_convert_images():
    # Get the list of uploaded files from the request
    files = request.files.getlist('images')
    
    # Call the controller function with the files
    predictions = convert_images(files)
    
    print(predictions)
    return predictions

if __name__ == '__main__':
    app.run(debug=True)
