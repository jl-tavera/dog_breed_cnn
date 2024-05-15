from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("./models/SGD-0.01-0.99-20epochs.keras")
def class_names_txt(filepath):
    """
    Read a text file and save each line as an element in a list.

    Args:
    - filepath (str): The path to the text file.

    Returns:
    - lines (list): A list containing each line of the text file as an element.
    """
    names = []
    with open(filepath, 'r') as file:
        for line in file:
            # Remove newline characters and any trailing/leading whitespaces
            cleaned_line = line.strip()
            cleaned_line = cleaned_line.replace("'", "")
            cleaned_line = cleaned_line.replace(",", "")
            names.append(cleaned_line)
    return names

filepath = "Dataset/breeds.txt"  
CLASS_NAMES = class_names_txt(filepath) 


@app.get("/ping")
async def ping():
    return {"ping": "pong"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    return np.array(image)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image = tf.image.resize(image, [256, 256])
    img_batch = np.expand_dims(image, axis=0)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {"class": predicted_class, "confidence": float(confidence)}

if __name__ == "__main__":
    run(app, host="localhost", port=8000)