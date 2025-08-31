from langchain_groq import ChatGroq
from typing import Optional
from pydantic import BaseModel,EmailStr,Field

from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv("GROQ_API_KEY")

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key
)

class student(BaseModel):
    name:str="kinza"
    age:Optional[int]=None
    roll_no:int
    email:EmailStr
    cgpa:float = Field(gt=0 , lt=4, default=3,desciption="value representing cgpa of student")

new_student={"age":"21","roll_no":2,"email":"kinza@123.com"}
Student=student(**new_student)

student_dict = Student.model_dump()
print(student_dict['age'])
print(student_dict['cgpa'])