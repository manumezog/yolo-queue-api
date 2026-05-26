# Case Study: Building a Production-Ready Asynchronous Object Detection API

### 🚀 Scaling YOLOv8 Vision Inference Beyond Toy Scripts

**🔗 Links:** [GitHub Repository](https://github.com/manumezog/yolo-queue-api) | [Live API Docs App](https://manumezog-yolo-queue-api.hf.space/docs)

---

## 📌 Executive Summary
In computer vision development, moving a model from a local testing script (`model.predict()`) into a production cloud environment introduces major engineering bottlenecks. Deep learning inference is heavily CPU/GPU intensive. If multiple users send images simultaneously to a standard synchronous API, the server threads freeze, resulting in connection timeouts, memory exhaustion, and a terrible user experience.

To solve this, I designed and built a **Production-Ready Asynchronous Object Detection API** using **FastAPI** and **YOLOv8**. The system treats image inference as a decoupled background task, managing multi-user load sequentially through a managed queue while protecting cloud hardware assets from resource exhaustion.

---

## 🏛️ System Architecture

The application decouples the client request from the heavy model execution loop using an asynchronous polling pattern:



1. **The Ingestion Shield (`POST /predict/`):** Receives the multi-part file upload. Validates MIME headers instantly to block invalid formats (e.g., PDFs, text files) before they waste processing memory.
2. **The Asynchronous Queue:** Assigns a unique UUID `task_id`, pushes the binary image stream into a sequential processing background thread, and immediately returns a `201 Accepted` status to the client.
3. **The Worker Loop:** Decodes the image stream dynamically using OpenCV matrix manipulations (`cv2.imdecode`), feeds it into a pre-loaded YOLOv8 instance, plots the bounding box annotations, and converts the output to an efficient Base64 image string.
4. **The Polling State Machine (`GET /result/{task_id}`):** The client polls this endpoint concurrently. The server reads the internal task state machine (`pending` -> `processing` -> `completed` / `error`) without locking the main thread.

---

## 🧹 Engineering Challenges & Production Hardening

### 1. Eliminating Container OOM (Out Of Memory) Crashes
**The Problem:** In-memory state tracking (`tasks_db = {}`) hoards massive Base64 string structures in the cloud instance's RAM. Under a heavy automated stress test, memory scales linearly until the OS kernel forcefully shuts down the container via an **OOM killer** event.

**The Solution:** I engineered an independent garbage-collection cycle utilizing a low-overhead **Daemon Thread** (`threading.Thread(..., daemon=True)`). 
* The thread wakes up automatically every 60 seconds.
* It evaluates the mathematical delta between the current timestamp and task completion records: $\Delta t = t_{current} - t_{updated\_at}$.
* If $\Delta t > 300\text{ seconds}$ (5 minutes), the task context is cleanly scrubbed from RAM.
* Because it is a *daemon thread*, its lifecycle is strictly bound to the master FastAPI process, completely eliminating zombie memory leaks upon server reboots.

### 2. Defensive Exception Interception
OpenCV natively returns a quiet `None` object if an image payload is corrupted or structurally broken, which traditionally forces an unhandled crash inside the deep learning pipeline. I introduced an intermediate array verification layer:
```python
nparr = np.frombuffer(file_bytes, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

if img is None:
    raise ValueError("Failed to decode image: corrupted or invalid format")

```

This intercepts systemic issues early, seamlessly translating low-level C++ library failures into standard HTTP error responses.

### 3.🏎️ Stress Test Performance Metrics
To stress-test the architectural limits of the cloud deployment, I built an asynchronous multi-connection test tool using Python's asyncio and httpx.

The script fires parallel high-resolution image payloads concurrently to test queue scheduling and response latencies:

Plaintext
--- INICIANDO ENVÍO EN PARALELO ---
🚀 [Tarea 1] Enviando imagen a la cola...
🚀 [Tarea 2] Enviando imagen a la cola...
🚀 [Tarea 3] Enviando imagen a la cola...
📥 [Tarea 2] Aceptada. ID: 185f9844-...
📥 [Tarea 1] Aceptada. ID: 93445b71-...
📥 [Tarea 3] Aceptada. ID: 8fb6c623-...

--- INICIANDO MONITOREO EN PARALELO ---
🔄 [Tarea 3] Verificando... Estado: processing
🔄 [Tarea 2] Verificando... Estado: processing
✨ [Tarea 2] ¡Éxito! Guardada como stress_output_2_185f9844.jpg
✨ [Tarea 1] ¡Éxito! Guardada como stress_output_1_93445b71.jpg
✨ [Tarea 3] ¡Éxito! Guardada como stress_output_3_8fb6c623.jpg

### 4.⏱️ Tiempo total de la prueba con alta concurrencia: 9.01 segundos
Key Takeaway: Even under spikes in concurrent traffic, the API successfully handles multiple simultaneous requests. It enqueues workloads safely, avoids blocking incoming connections, and executes them sequentially without any drop in server stability.

## 🛠️ Tech Stack & Key Tooling
Core Framework: FastAPI (ASGI Python web server framework)

Computer Vision: OpenCV (image streaming, byte-matrix transformations) & Ultralytics YOLOv8 (Inference engine)

Concurrency: Asyncio & HTTPX (for asynchronous local stress testing loops)

Containerization & Cloud Dev: Docker (custom multi-stage Linux runtime environment built to correctly link low-level system graphics dependencies like libGL.so.1), deployed to Hugging Face Spaces.
