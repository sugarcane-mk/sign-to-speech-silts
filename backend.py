from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import shutil
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/process")
async def process_video(video: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Run the shell script
    subprocess.run(["bash", "main_run.sh", video_path])

    # Get predicted text
    with open("output.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()

    return JSONResponse(content={"text": text})

@app.get("/audio/test.wav")
async def get_audio():
    return FileResponse("test.wav", media_type="audio/wav")
