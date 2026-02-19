import requests
import io
import time
import os
from PIL import Image

# Configuration
# BASE_URL = "http://localhost:8000"
# Pointing to the deployed server
BASE_URL = "http://122.169.42.110:1415"
# REPLACE THIS with the actual path to an image on your laptop
REAL_IMAGE_PATH = r"C:\Users\sachi\Desktop\HKTASKMODEL\dataset\images\testimage.jpeg"

def create_dummy_image(color="blue"):
    """Creates a simple in-memory image for testing."""
    img = Image.new('RGB', (200, 200), color=color)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()

def test_health():
    """Tests the /health endpoint."""
    print(f"Testing GET {BASE_URL}/health ...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")
    except Exception as e:
        print(f"Failed: {e}")

def test_index_image():
    """Tests the /index endpoint to add a reference image."""
    print(f"Testing POST {BASE_URL}/index ...")
    
    image_bytes = create_dummy_image(color="green")
    
    files = {
        'file': ('clean_lobby.jpg', image_bytes, 'image/jpeg')
    }
    # Note: 'status' must match the Enum in models.py (Clean/Dirty)
    data = {
        'status': 'Clean',
        'description': 'A perfectly clean green floor example.'
    }
    
    try:
        response = requests.post(f"{BASE_URL}/index", files=files, data=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")
    except Exception as e:
        print(f"Failed: {e}")

def test_evaluate_images():
    """Tests the /evaluate endpoint with multiple images."""
    print(f"Testing POST {BASE_URL}/evaluate ...")
    
    img1 = create_dummy_image(color="red")
    img2 = create_dummy_image(color="darkred")
    
    # FastAPI expects a list of files for the 'files' key
    files = [
        ('files', ('spill_1.jpg', img1, 'image/jpeg')),
        ('files', ('spill_2.jpg', img2, 'image/jpeg'))
    ]
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/evaluate", files=files)
        print(f"Status: {response.status_code}")
        print(f"Time Taken: {time.time() - start_time:.2f}s")
        try:
            print(f"Response: {response.json()}\n")
        except:
            print(f"Response Text: {response.text[:200]}...\n")
    except Exception as e:
        print(f"Failed: {e}")

def test_real_image(image_path):
    """Tests the /evaluate endpoint with a real local file."""
    print(f"Testing Real Image: {image_path} ...")
    
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        print("Please update REAL_IMAGE_PATH in test_app.py")
        return

    try:
        with open(image_path, "rb") as f:
            # 'files' key matches the FastAPI endpoint argument
            files = [('files', (os.path.basename(image_path), f, 'image/jpeg'))]
            
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/evaluate", files=files)
            print(f"Status: {response.status_code}")
            print(f"Time Taken: {time.time() - start_time:.2f}s")
            try:
                print(f"Response: {response.json()}\n")
            except:
                print(f"Response Text: {response.text[:200]}...\n")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    # test_health()
    # test_index_image()
    # test_evaluate_images()
    # test_real_image(REAL_IMAGE_PATH)
    test_evaluate_images()
