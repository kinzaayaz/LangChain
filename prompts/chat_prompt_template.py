from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os

api_key=os.getenv("GROQ_API_KEY")
model=ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key
)

chat_template=ChatPromptTemplate.from_messages([
    ("system","you are a great {domain} expert."),
    ("user","tell me about {topic}.")
])
prompt=chat_template.format_messages(domain="AI", topic="ROBOTICS")
result=model.invoke(prompt)
print("AI",result.content)