from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os

api_key=os.getenv("GROQ_API_KEY")

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key
)

message=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="tell me about langchain")
]
result = model.invoke(message)
message.append(AIMessage(content=result.content))
print(message)