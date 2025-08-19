import requests

url = "http://127.0.0.1:5000/upload"
image_path = r"C:\Users\akash\OneDrive\Documents\Desktop\python backend file\test.jpg.jpg"

with open(image_path, "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)

print(response.json())
