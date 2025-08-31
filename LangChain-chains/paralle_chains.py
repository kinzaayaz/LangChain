from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
import os

# Load API Key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Models
model1=ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key
)

model2=ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=api_key
)

prompt1=PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="generate 5 question answers from given text.\n{text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="merge the give notes and quiz into single document.\n notes {notes} and quiz {quiz} ",
    input_variables=['notes',"quiz"]
)

parser= StrOutputParser()

parallel_chain = RunnableParallel({
    "notes" : prompt1 |model1 | parser,
    "quiz" : prompt2 | model2 |parser
})

merge_chain = prompt3 | model2 | parser

chain =parallel_chain | merge_chain
result=chain.invoke("""
Artificial Intelligence (AI) is a branch of computer science that focuses on creating machines and software capable of performing tasks that normally require human intelligence. These tasks include learning from data, reasoning, problem-solving, understanding natural language, recognizing images and speech, and making decisions.

AI can be broadly divided into two types: Narrow AI and General AI. Narrow AI is designed to handle specific tasks, such as recommendation systems, chatbots, or image recognition tools. It is the most common form of AI used today. General AI, on the other hand, refers to a system that could perform any intellectual task a human can do, but it remains a theoretical concept.

Several techniques power AI, including machine learning, where systems learn patterns from large datasets; deep learning, which uses neural networks with many layers to process complex data; and natural language processing (NLP), which allows machines to understand and generate human language.

AI is widely used across industries: in healthcare for disease prediction and drug discovery, in finance for fraud detection and trading algorithms, in education for personalized learning, and in everyday applications such as digital assistants (like Siri and Alexa), recommendation engines (like Netflix and YouTube), and autonomous vehicles.

While AI brings many benefits like efficiency, automation, and new opportunities, it also raises concerns. Challenges include ethical issues, job displacement due to automation, data privacy, bias in algorithms, and the need for proper regulations. Researchers and policymakers are working to ensure that AI is developed responsibly and used for the benefit of society.
""")
print(result)
chain.get_graph().print_ascii()