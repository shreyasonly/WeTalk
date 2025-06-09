import boto3
import bcrypt
import uuid
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
import jwt


dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
user_table = dynamodb.Table('Users_table')


class RegisterUser(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def register_user(user: RegisterUser):
    
    result = user_table.scan(
        FilterExpression="email = :e",
        ExpressionAttributeValues={":e": user.email}
    )

    if result.get("Items"):
        raise HTTPException(status_code=400, detail="User already exists.")

    user_id = str(uuid.uuid4())
    password_hash = hash_password(user.password)

    user_table.put_item(Item={
        "user_id": user_id,
        "email": user.email,
        "password_hash": password_hash,
        "created_at": datetime.utcnow().isoformat()
    })

    return {"message": "User registered successfully.", "user_id": user_id}

# Secret key and algorithm for JWT
SECRET_KEY = "mysecretkey"  
ALGORITHM = "HS256"

# OAuth2 scheme for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Dependency to verify the current user via JWT token
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Login function
def login_user(login: LoginRequest):
    result = user_table.scan(
        FilterExpression="email = :e",
        ExpressionAttributeValues={":e": login.email}
    )
    items = result.get("Items", [])
    if len(items) == 0:
        raise HTTPException(status_code=404, detail="User not found.")
    user = items[0]
    stored_hash = user["password_hash"]
    if not bcrypt.checkpw(login.password.encode('utf-8'), stored_hash.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    # Generate JWT token
    payload = {
        "sub": user["user_id"],
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return {"token": token, "user_id": user["user_id"]}