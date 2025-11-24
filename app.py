from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from fastapi.responses import JSONResponse
import cv2
import numpy as np

app = FastAPI()

# Habilita CORS global
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga el modelo (Render lo cargará desde esta ruta)
model = load_model('reconocimiento-rostro/1')

# ⚠️ Orden ALFABÉTICO, igual que el LabelEncoder
listaPersonas = ['Daniel', 'Erick', 'Josue', 'Luis', 'William']

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    try:
        # Leer la imagen enviada
        contents = await image.read()
        
        # Convertir bytes → imagen (escala de grises)
        image_array = cv2.imdecode(
            np.frombuffer(contents, np.uint8), 
            cv2.IMREAD_GRAYSCALE
        )

        # Redimensionar a lo que espera el modelo
        image_array = cv2.resize(image_array, (150, 150))

        # Normalizar
        image_array = image_array / 255.0

        # Predicción
        prediction = model.predict(
            np.array([image_array]).reshape(-1, 150, 150, 1)
        )

        predicted_label = int(np.argmax(prediction))
        predicted_name = listaPersonas[predicted_label]

        return JSONResponse({
            "predicted_label": predicted_label,
            "predicted_name": predicted_name
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Render no usa esto, pero te sirve localmente.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)
