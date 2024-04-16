from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import numpy as np
from io import BytesIO
from pydub import AudioSegment
import tensorflow as tf
from keras.layers import TFSMLayer

app = FastAPI()

MODEL = TFSMLayer("C:/Users/olish/DrumClassifier/models/2", call_endpoint="serving_default")
CLASS_NAMES = [
 'bass',
 'bass_snare',
 'bass_toms',
 'overheads',
 'overheads_bass',
 'overheads_snare',
 'overheads_toms',
 'snare',
 'snare_toms',
 'toms'
]
@app.get("/ping")
async def ping():
    return "Hello"

def read_file_as_audio(data) -> np.ndarray:
    audio_segment = AudioSegment.from_file(BytesIO(data))
    audio_array = np.array(audio_segment.get_array_of_samples())
    return audio_array
@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    audio = read_file_as_audio(await file.read())

    audio_batch = np.expand_dims(audio,0)

    MODEL.predict(audio_batch)
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)