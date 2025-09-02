from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
import os

load_dotenv()

groq_api_key= os.getenv("GROQ_API_KEY")

def word_count(text):
    return len(text.split())

prompt=PromptTemplate(
    template="write a joke on {topic}",
    input_variables=['topic']
)

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key
)

parser=StrOutputParser()

joke_gen_chain=RunnableSequence(prompt,model,parser)

parallel_chain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "word_count":RunnableLambda(word_count)
})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)
result=final_chain.invoke({"topic":"Study"})
print(result['joke'])
print(result['word_count'])