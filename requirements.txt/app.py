from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
import numpy as np

app = Flask(_name_)

def lightweight_predict(img_array):
    # Simple classifier: checks average color channels
    mean_color = np.mean(img_array, axis=(0, 1))
    if mean_color[2] > 150:
        label = 'mobile phone'
    elif mean_color > 120:
        label = 'laptop'
    else:
        label = 'home theater'
    return label

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
    "home theater": {
        "product": "Home Theater",
        "material": "Plastic/Metal",
        "type": "Audio Equipment",
        "price": 25000,
        "usage": "Entertainment",
        "lifespan": "4-6 years",
        "amazon": "https://www.amazon.in/s?k=home+theater"
    },
    "mobile phone": {
        "product": "Mobile Phone",
        "material": "Glass/Metal/Plastic",
        "type": "Electronics",
        "price": 20000,
        "usage": "Communication",
        "lifespan": "2-3 years",
        "amazon": "https://www.amazon.in/s?k=cell+phone"
    }
}

@app.route('/')
def index():
    return "Product Quality API is running!"

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image uploaded'}), 400
        img = Image.open(file.stream).convert("RGB").resize((128, 128))
        arr = np.array(img)
        label = lightweight_predict(arr)
        response = product_map.get(label, {
            "product": label,
            "material": "Unknown",
            "type": "Unknown",
            "price": 0,
            "usage": "Unknown",
            "lifespan": "Unknown",
            "amazon": "https://www.amazon.in"
        })
        return jsonify(response)
    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image format"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=10000)