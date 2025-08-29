from langchain_groq import ChatGroq
# from langchain_prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

api_key=os.getenv("GROQ_API_KEY")

model=ChatGroq(
    model='llama-3.3-70b-versatile',
    groq_api_key=api_key
)
system_prompt="""
you are a helpful assistant
you only answer questions related to robotics.
if the user ask about anything other than robots, than rply:
"sorry i can only answer question about robotics"
"""

chat_template=ChatPromptTemplate.from_messages([
    ("system",system_prompt), 
    ("human","{user_query}")
])

while True:
    user_q=input("You: ")
    if user_q == 'exit':
        break
    msgs=chat_template.format_messages(user_query=user_q)
    result=model.invoke(msgs)
    print("AI",result.content)