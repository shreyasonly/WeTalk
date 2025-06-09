from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from auth import RegisterUser, register_user, LoginRequest, login_user, get_current_user,hash_password
from pydantic import BaseModel, EmailStr
import os
import boto3
import shutil
import whisper
from transcription_utils import (
    convert_to_wav, denoise_audio, split_audio,
    process_transcription, correct_grammar, transcribe_gemini
)
from db import save_transcription
from datetime import datetime

app = FastAPI()
os.makedirs("temp", exist_ok=True)

dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
user_table = dynamodb.Table('Users_table')


@app.post("/transcribe")
async def transcribe_audio(
    user_id: str = Form(...),
    model: str = Form(...),  # 'whisper' or 'gemini'
    whisper_model_size: str = Form("tiny"),
    file: UploadFile = File(...)
):
    try:
        temp_path = os.path.join("temp", file.filename)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        temp_wav = os.path.join("temp", "converted.wav")
        denoised_wav = os.path.join("temp", "denoised.wav")

        convert_to_wav(temp_path, temp_wav)
        denoise_audio(temp_wav, denoised_wav)
        segments = split_audio(denoised_wav)

        final_text = ""

        if model == "whisper":
            model = whisper.load_model(whisper_model_size)
            for seg in segments:
                result = model.transcribe(seg)
                cleaned = process_transcription(result["text"])
                final_text += correct_grammar(cleaned) + "\n"
                os.remove(seg)

        elif model == "gemini":
            for seg in segments:
                raw = transcribe_gemini(seg)
                cleaned = process_transcription(raw)
                final_text += correct_grammar(cleaned) + "\n"
                os.remove(seg)

        save_transcription(user_id, file.filename, model, final_text.strip())

        return {"user_id": user_id, "model": model, "transcription": final_text.strip()}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        for path in [temp_path, temp_wav, denoised_wav]:
            if os.path.exists(path):
                os.remove(path)


@app.post("/register")
def register(user: RegisterUser):
    return register_user(user)


@app.post("/login")
def login(login: LoginRequest):
    return login_user(login)


class UpdateUser(BaseModel):
    email: EmailStr | None = None
    password: str | None = None


@app.get("/users/{user_id}")
def get_user(user_id: str, current_user: str = Depends(get_current_user)):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="Not authorized")
    response = user_table.get_item(Key={"user_id": user_id})
    item = response.get("Item")
    if not item:
        raise HTTPException(status_code=404, detail="User not found")
    return item

# PUT /users/{user_id} 
@app.put("/users/{user_id}")
def update_user(user_id: str, update: UpdateUser, current_user: str = Depends(get_current_user)):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="Not authorized")
    response = user_table.get_item(Key={"user_id": user_id})
    if not response.get("Item"):
        raise HTTPException(status_code=404, detail="User not found")
    update_data = {}
    if update.email:
        update_data["email"] = update.email
    if update.password:
        update_data["password_hash"] = hash_password(update.password)
    if update_data:
        user_table.update_item(
            Key={"user_id": user_id},
            UpdateExpression="SET " + ", ".join(f"{k} = :{k}" for k in update_data),
            ExpressionAttributeValues={f":{k}": v for k, v in update_data.items()}
        )
    return {"message": "User updated successfully"}

#DELETE /users/{user_id} 
@app.delete("/users/{user_id}")
def delete_user(user_id: str, current_user: str = Depends(get_current_user)):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="Not authorized")
    response = user_table.get_item(Key={"user_id": user_id})
    if not response.get("Item"):
        raise HTTPException(status_code=404, detail="User not found")
    user_table.delete_item(Key={"user_id": user_id})
    return {"message": "User deleted successfully"}