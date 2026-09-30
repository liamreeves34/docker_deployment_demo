from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
from models.predict import DigitPredictor

app = FastAPI()
predictor = DigitPredictor()


@app.get("/health")
def get_health():
    return {"status": "healthy"}

#need to define what happens if an error occurs at every step
#this enpoint needs to: recieve an image as a file, open it as a PIL image via Image.open(io.bytes(file))

@app.post("/predict/")
async def create_upload_file(file: UploadFile=File(...)):
    # files are converted to raw bytes when you upload a file to the api, which is an IO operation that is slow
    # IO is slow, so we use the keyword "await" to specify we can complete other tasks while waiting for file read
    raw_bytes = await file.read()              
    
    # Pillow expects a "file-like" object not raw bytes, so io.BytesIO(raw_bytes) converts to a "file-like" object
    # that Pillow can read without the image having to be on disk
    try:
        image = Image.open(io.BytesIO(raw_bytes)) 
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="File is not a valid image"
        )
    
    # We need to check if a PNG was uploaded since our predict method only uses 
    if image.format != "PNG":
        raise HTTPException(
            status_code=400,
            detail=f"Only PNG's allowed. You uploaded a {image.format}"
        )
    prediction, confidence = predictor.predict(image)              
                                               
    return {"prediction": prediction, "confidence": confidence}
    