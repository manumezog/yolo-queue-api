import requests
import time
import base64

# 🌐 Cloud API Configuration
base_url = "https://manumezog-yolo-queue-api.hf.space"
predict_url = f"{base_url}/predict/"
file_path = r"images/KakaoTalk_20241024_185211902_04_jpg.rf.e911b2c055b611c50129f7ac83994d96.jpg"
print("Sending image to queue...")

# 📥 Open file and send the POST request all within the same safe block
with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "image/jpeg")}
    task_data = requests.post(predict_url, files=files).json()

# 🔑 Extract the unique task identifier
task_id = task_data.get("task_id")
print(f"Task accepted! ID assigned: {task_id}")

# 🔗 Dynamically construct the result URL using the cloud base URL
result_url = f"{base_url}/result/{task_id}"

# 🔄 Polling loop
while True:
    status_response = requests.get(result_url).json()
    status = status_response.get("status")
    
    print(f"Checking server... Status is: {status}")
    
    if status == "completed":
        # Extract the Base64 image string from the result object
        header, base64_data = status_response["result"].split(";base64,")
        image_bytes = base64.b64decode(base64_data)
        
        # 📂 Save the output with the unique task ID
        filename = f"queue_output_{task_id}.jpg"
        with open(filename, "wb") as f:
            f.write(image_bytes)
            
        print(f"Success! Labeled image saved as {filename}")
        break
        
    elif status == "error":
        print(f"Error during processing: {status_response.get('message')}")
        break
        
    time.sleep(1)  # Wait 1 second before checking again