from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.database import SessionLocal
from models.user import User
from pydantic import BaseModel
import hashlib

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.user_name == request.username, User.pass_word == request.password).first()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        return {"message": "Login successful", "username": user.user_name, "role": user.role}
    except HTTPException:
        raise HTTPException(status_code=401, detail="Invalid username or password")
