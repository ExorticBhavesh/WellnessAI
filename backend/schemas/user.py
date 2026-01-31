from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    age: int
    gender: str
    height: float
    weight: float
