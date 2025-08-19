import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU (CUDA)

from flask import Flask, request, jsonify
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load MobileNet model once on startup
model = MobileNet(weights='imagenet')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            print("No image provided in request.files")
            return jsonify({'error': 'No image provided'}), 400

        img_file = request.files['image']
        print(f"Received file: {img_file.filename}")

        try:
            img = image.load_img(img_file, target_size=(224, 224))
        except Exception as e:
            print("Error loading image:", str(e))
            return jsonify({'error': f"Image load error: {str(e)}"}), 400

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        decoded = decode_predictions(preds, top=3)[0]

        results = [{'label': label, 'description': desc, 'probability': float(prob)} for (label, desc, prob) in decoded]

        return jsonify({'filename': img_file.filename, 'predictions': results})

    except Exception as e:
        print("General error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
