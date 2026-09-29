from fastapi import FastAPI, Path, File, UploadFile
from PIL import Image
import io
from models.predict import DigitPredictor

app = FastAPI()

@app.get("/health")
def get_health():
    return {"status": "healthy"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile=File(...)): #File needs to be included as parameter of query or error
    return {"filename": file.filename}
    
