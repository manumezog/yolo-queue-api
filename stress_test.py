import asyncio
import httpx
import time
import base64

# 🌐 Configuración de la API en la nube
base_url = "https://manumezog-yolo-queue-api.hf.space"
predict_url = f"{base_url}/predict/"
file_path = r"images\-1-_jpeg.rf.ab4ad7f0f252559084db6bd9d72d2ee0.jpg"

async def send_image(client, task_number):
    """Envía la imagen a la API y devuelve el task_id asignado."""
    print(f"🚀 [Tarea {task_number}] Enviando imagen a la cola...")
    
    # Abrimos el archivo para cada petición
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "image/jpeg")}
        response = await client.post(predict_url, files=files)
        task_data = response.json()
        
    task_id = task_data.get("task_id")
    print(f"📥 [Tarea {task_number}] Aceptada. ID: {task_id}")
    return task_id

async def poll_task(client, task_id, task_number):
    """Consulta el estado de una tarea específica hasta que se completa o falla."""
    result_url = f"{base_url}/result/{task_id}"
    
    while True:
        response = await client.get(result_url)
        status_response = response.json()
        status = status_response.get("status")
        
        print(f"🔄 [Tarea {task_number}] Verificando... Estado: {status}")
        
        if status == "completed":
            # Extraer y decodificar la imagen final
            header, base64_data = status_response["result"].split(";base64,")
            image_bytes = base64.b64decode(base64_data)
            
            filename = f"stress_output_{task_number}_{task_id[:8]}.jpg"
            with open(filename, "wb") as f:
                f.write(image_bytes)
                
            print(f"✨ [Tarea {task_number}] ¡Éxito! Guardada como {filename}")
            break
            
        elif status == "error":
            print(f"❌ [Tarea {task_number}] Error: {status_response.get('message')}")
            break
            
        await asyncio.sleep(1) # Espera asíncrona de 1 segundo

async def main():
    # Creamos un único cliente HTTP asíncrono para todas las peticiones
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Disparamos 3 peticiones de predicción en paralelo
        print("--- INICIANDO ENVÍO EN PARALELO ---")
        envios = [send_image(client, i) for i in range(1, 4)]
        task_ids = await asyncio.gather(*envios)
        
        print("\n--- INICIANDO MONITOREO EN PARALELO ---")
        # 2. Monitoreamos el estado de las 3 tareas al mismo tiempo
        monitoreos = [poll_task(client, task_ids[i], i+1) for i in range(3)]
        await asyncio.gather(*monitoreos)

# Ejecutamos el bucle de eventos asíncrono
if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    print(f"\n⏱️ Tiempo total de la prueba: {time.time() - start_time:.2f} segundos")