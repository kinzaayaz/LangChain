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
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


model_structure = model.with_structured_output(json_schema)

result = model_structure.invoke("""Emily Johnson writes that the smartphone delivers extremely fast performance with smooth multitasking, a brilliant 4K OLED display with vibrant colors, long-lasting battery life even with heavy usage, a top-notch camera system with multiple shooting modes, and premium build quality with a sleek design. However, she notes that it is very expensive compared to competitors, large and difficult to use one-handed, heavier than previous models, and that software updates occasionally introduce bugs. Overall, after using it for a month, Emily feels that despite the drawbacks, the device provides an excellent experience for tech enthusiasts who want top performance and features, making it highly recommendable.""")

print(result)