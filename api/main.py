from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Create the FastAPI app
app = FastAPI()

# Allow requests from the frontend
origins = [
    "http://localhost",
    "http://localhost:3000",
]

# Add the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def class_names_txt(filepath):
    '''	
    Read a text file and save each line as an element in a list.
    '''
    names = []
    with open(filepath, 'r') as file:
        for line in file:
            # Remove newline characters and any trailing/leading whitespaces
            cleaned_line = line.strip()
            cleaned_line = cleaned_line.replace("'", "")
            cleaned_line = cleaned_line.replace(",", "")
            names.append(cleaned_line)
    return names

# Define the path to the breeds.txt file
filepath = "./Dataset/breeds.txt"  
# Load the model
CLASS_NAMES = class_names_txt(filepath) 
MODEL = tf.keras.models.load_model("./models/SGD-0.01-0.99-20epochs.keras")


def read_file_as_image(data) -> np.ndarray:
    '''
    Read the image file as a numpy array
    '''
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))

    return np.array(image) / 255.0


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    '''
    Predicts the class of the image
    '''
    
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {"class": predicted_class, "confidence": float(confidence)}

if __name__ == "__main__":
    run(app, host="localhost", port=8000)