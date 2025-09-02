from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key= os.getenv("GROQ_API_KEY")

prompt1= PromptTemplate(
    template= "generate a short tweet about {topic}.",
    input_variables=['topic']
)
prompt2= PromptTemplate(
    template= "generate a short post about {topic}.",
    input_variables=['topic']
)

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key
)

parser=StrOutputParser()

parallel_chains= RunnableParallel({
    "tweet":RunnableSequence(prompt1,model,parser),
    "linkdin": RunnableSequence(prompt2,model,parser)
})

result=parallel_chains.invoke({"topic":"ML"})
# print(result)
print("tweet: ",result['tweet'])