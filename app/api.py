from fastapi import FastAPI, Request
from fastapi import Body
from app.schemas import Text
from pathlib import Path
import uvicorn
from CDS_Classifier import predict

app = FastAPI()

# BASE_DIR = Path(__file__).parent.parent
# predict_file = Path(BASE_DIR,"CDS_Classifier","predict.py")
# print(predict_file)

@app.get("/")
def index():
    response = {'Welcome to the app'}
    return response

@app.post("/predict")
def prediction(new_transcript: Text):
    print(new_transcript.transcript)
    print(type(new_transcript.transcript))
    zero_or_one = predict.prediction(new_transcript.transcript)
    print(type)
    print(zero_or_one)
    return {"message": str(zero_or_one)}

if __name__ == "__main__":
     uvicorn.run(app,host="0.0.0.0",port=8000)