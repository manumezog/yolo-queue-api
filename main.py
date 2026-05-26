import uuid
import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from ultralytics import YOLO
from pydantic import BaseModel
from typing import Optional, Any
import textwrap
import threading
import time

# Schema for POST /predict/ response
class TaskSubmissionResponse(BaseModel):
    task_id: str
    status: str

# Schema for GET /result/{task_id} response
class TaskResultResponse(BaseModel):
    status: str
    # 'Any' allows this to hold the image string or an error message
    result: Optional[Any] = None 
    message: Optional[str] = None

app = FastAPI(
    title="🚀 YOLOv8 Asynchronous Object Detection API",
    description=textwrap.dedent("""
        Welcome to the YOLOv8 Object Detection API!
        This API processes heavy image detection tasks using an Asynchronous Queue system.
        
        🚦 **Workflow Steps:**
        1. Submit an image to `/predict/` to receive a unique `task_id`.
        2. Poll the `/result/{task_id}` endpoint to check the processing status.
        3. Retrieve the final Base64 annotated image once the status is `completed`.
        
        🧹 **Memory Management Notice:**
        To keep the server's RAM safe, completed or failed tasks are stored in memory for a maximum of **5 minutes** (300 seconds). 
        After this window, tasks are automatically purged. If you query an expired ID, the server will respond with a `404 Not Found` error.
    """),
    version="4.20",
    contact={
        "name": "Developed with love by Manu Mezo for learning purposes",
        "url": "https://github.com/manumezog",
    }
)

# Load the model once into memory
model = YOLO("yolov8n.pt") 

# In-memory database to store task statuses and results
tasks_db = {}

def fastapi_queue_cleaner():
    """Bucle que revisa tasks_db y elimina tareas con más de 5 minutos de antigüedad."""
    print("🧹 Filtro de limpieza de RAM iniciado.")
    while True:
        current_time = time.time()
        # Usamos una lista de llaves para evitar errores de cambio de tamaño durante la iteración
        expired_tasks = []
        
        for task_id, task_info in list(tasks_db.items()):
            # Revisamos si la tarea tiene una marca de tiempo
            updated_at = task_info.get("updated_at", 0)
            # 300 segundos = 5 minutos
            if current_time - updated_at > 300:
                expired_tasks.append(task_id)
                
        for task_id in expired_tasks:
            del tasks_db[task_id]
            print(f"🗑️ Tarea expirada eliminada de RAM: {task_id}")
            
        # El limpiador revisa la memoria cada 60 segundos
        time.sleep(60)

# Iniciamos el hilo de limpieza inmediatamente al arrancar el servidor
cleaner_thread = threading.Thread(target=fastapi_queue_cleaner, daemon=True)
cleaner_thread.start()

def process_yolo_task(task_id: str, file_bytes: bytes):
    """Heavy YOLO task running sequentially in the background."""
    tasks_db[task_id] = {"status": "processing"}
    
    try:
        # Decode image
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Validate image was decoded successfully
        if img is None:
            raise ValueError("Failed to decode image: corrupted or invalid format")

        # Run inference
        results = model(img)[0]
        
        # Annotate image
        annotated_img = results.plot()
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Save results to our temporary DB
        tasks_db[task_id] = {
            "status": "completed",
            "result": f"data:image/jpeg;base64,{img_base64}",
            "updated_at": time.time()  # ⏱️ Registra el momento de finalización
        }
    except Exception as e:
        tasks_db[task_id] = {
            "status": "error",
            "message": str(e),
            "updated_at": time.time()  # ⏱️ Registra el momento del fallo
        }

@app.post("/predict/", tags=["🎯 Detection Queue"], response_model=TaskSubmissionResponse)
async def predict_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 🛡️ Validation Shield: Only allow JPEG and PNG
    ALLOWED_TYPES = ["image/jpeg", "image/png", "image/jpg"]
    
    if file.content_type not in ALLOWED_TYPES:
        # We can return a 400 Bad Request status code or a custom message
        return {"task_id": "none", "status": f"Rejected: Files of type {file.content_type} are not supported."}
    
    # (The rest of your successful queue logic continues safely below...)
    task_id = str(uuid.uuid4())
    file_bytes = await file.read()
    
    tasks_db[task_id] = {"status": "queued"}
    
    # Push to background queue
    background_tasks.add_task(process_yolo_task, task_id, file_bytes)
    
    return {"task_id": task_id, "status": "queued"}

@app.get("/result/{task_id}",tags=["📊 Task Tracking"], response_model=TaskResultResponse)
async def get_result(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task expired or not found")
    return tasks_db[task_id]