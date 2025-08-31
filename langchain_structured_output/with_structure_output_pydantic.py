from langchain_groq import ChatGroq
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv("GROQ_API_KEY")

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key
)

#schema
class schema(BaseModel):
    summary : str = Field(desciption="A brief summary of review")
    sentiment : Literal['pos','neg'] = Field(desciption="sentiment of review")
    name: Optional[str]=Field(default=None , desciption='name of writer')
    pros : Optional[list[str]]=Field(default=None , desciption="list all pros if present in review")
    cons : Optional[list[str]]=Field(default=None , desciption="list all cons if present in review")

model_structure = model.with_structured_output(schema)

result = model_structure.invoke("""Emily Johnson writes that the smartphone delivers extremely fast performance with smooth multitasking, a brilliant 4K OLED display with vibrant colors, long-lasting battery life even with heavy usage, a top-notch camera system with multiple shooting modes, and premium build quality with a sleek design. However, she notes that it is very expensive compared to competitors, large and difficult to use one-handed, heavier than previous models, and that software updates occasionally introduce bugs. Overall, after using it for a month, Emily feels that despite the drawbacks, the device provides an excellent experience for tech enthusiasts who want top performance and features, making it highly recommendable.""")

print("summary : ",result.summary)
print("sentiment: ",result.sentiment)
print("name: ",result.name)