from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
load_dotenv()

api_key=os.getenv("GROQ_API_KEY")

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key
)

system_prompt="""
You are a medical expert. 
only answer the questions related to medical field . answer with short description.
Answer the user's questions in a friendly and polite manner. 
If someone asks anything outside the medical field, reply: 
"I can't assist you. Please ask something related to the any medical field."
"""

chat_template=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human","{user_query}")
])
chat_history=[SystemMessage(content=system_prompt)]
while True:
    user_input=input("you: ")
    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)
print(chat_history)