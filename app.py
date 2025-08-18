from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import traceback
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

app = Flask(__name__)

# Load MobileNetV2 model once on startup
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Expanded product mapping dictionary
product_map = {
    "laptop": {
        "product": "Laptop",
        "material": "Metal/Plastic",
        "type": "Electronics",
        "price": 60000,
        "usage": "Computing",
        "lifespan": "3-5 years",
        "amazon": "https://www.amazon.in/s?k=laptop"
    },
    "backpack": {
        "product": "Backpack",
        "material": "Fabric/Plastic",
        "type": "Bag",
        "price": 800,
        "usage": "Carry stuff",
        "lifespan": "2-4 years",
        "amazon": "https://www.amazon.in/s?k=backpack"
    },
    "home_theater": {
        "product": "Home Theater",
        "material": "Plastic/Metal",
        "type": "Audio Equipment",
        "price": 25000,
        "usage": "Audio, Entertainment",
        "lifespan": "4-6 years",
        "amazon": "https://www.amazon.in/s?k=home+theater"
    },
    "notebook": {
        "product": "Notebook Laptop",
        "material": "Metal/Plastic",
        "type": "Electronics",
        "price": 50000,
        "usage": "Work, Study",
        "lifespan": "3-5 years",
        "amazon": "https://www.amazon.in/s?k=notebook+laptop"
    },
    "headphones": {
        "product": "Headphones",
        "material": "Plastic/Leather",
        "type": "Electronics",
        "price": 1500,
        "usage": "Audio Listening",
        "lifespan": "2-3 years",
        "amazon": "https://www.amazon.in/s?k=headphones"
    },
    "cell_phone": {
        "product": "Mobile Phone",
        "material": "Glass/Metal/Plastic",
        "type": "Electronics",
        "price": 20000,
        "usage": "Communication",
        "lifespan": "2-3 years",
        "amazon": "https://www.amazon.in/s?k=cell+phone"
    }
}

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image uploaded'}), 400

        # Try to open the uploaded image
        try:
            img = Image.open(file.stream).convert("RGB").resize((224, 224))
        except UnidentifiedImageError:
            return jsonify({'error': 'Uploaded file is not a valid image'}), 400
        except Exception as e:
            print("Error opening image:", e)
            traceback.print_exc()
            return jsonify({'error': 'Failed to open image', 'detail': str(e)}), 500

        # Prepare image for MobileNetV2
        arr = np.array(img)
        arr = preprocess_input(arr)  # ✅ proper preprocessing
        arr = np.expand_dims(arr, axis=0)

        try:
            preds = model.predict(arr)
            print("Model raw preds shape:", preds.shape)
            decoded = decode_predictions(preds, top=1)
            print("Decoded output:", decoded)

            if not decoded or not decoded[0]:
                return jsonify({'error': 'Model could not identify image.'}), 400

            best = decoded[0][0]  # ('n03642806', 'laptop', 0.95)
            label = best[1]       # 'laptop'
            confidence = float(best[2])
            print(f"Predicted label: {label}, confidence: {confidence}")

        except Exception as e:
            print("Error during prediction:", e)
            traceback.print_exc()
            return jsonify({'error': 'Prediction failed', 'detail': str(e)}), 500

        # Map label to product info
        response = product_map.get(label.lower(), {
            "product": label,
            "material": "Unknown",
            "type": "Unknown",
            "price": 0,
            "usage": "Unknown",
            "lifespan": "Unknown",
            "amazon": "https://www.amazon.in"
        })
        response["confidence"] = confidence

        return jsonify(response)

    except Exception as e:
        print("General error in upload endpoint:", e)
        traceback.print_exc()
        return jsonify({'error': 'Server error', 'detail': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
