from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # For now, just return success without saving
    return jsonify({
        "filename": file.filename,
        "message": "File received successfully!"
    })

if __name__ == '__main__':
    app.run(debug=True)
