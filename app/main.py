from fastapi import FastAPI, Path, File, UploadFile
from PIL import Image
import io
from models.predict import DigitPredictor

app = FastAPI()

@app.get("/health")
def get_health():
    return {"status": "healthy"}

#need to define what happens if an error occurs at every step
#this enpoint needs to: recieve an image as a file, open it as a PIL image via Image.open(io.bytes(file))
#
@app.post("/predict/")
async def create_upload_file(file: UploadFile=File(...)): #File needs to be included as parameter of query or error
    predictor = DigitPredictor()
    
    
    return {"filename": file.filename}
    