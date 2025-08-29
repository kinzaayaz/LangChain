from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from dotenv import load_dotenv
load_dotenv()
import os

api_key=os.getenv("GROQ_API_KEY")

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key
)

chat_history=[
    SystemMessage(content="You are a helpgul assistant.")
]
while True:
    user_input = input("user: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)


