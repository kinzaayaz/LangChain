from langchain_text_splitters import RecursiveCharacterTextSplitter

text="""
Unlike the previous methods, semantic-based splitting actually considers the content of the text. While other approaches use document or text structure as proxies for semantic meaning, this method directly analyzes the text's semantics. There are several ways to implement this, but conceptually the approach is split text when there are significant changes in text meaning. As an example, we can use a sliding window approach to generate embeddings, and compare the embeddings to find significant differences:Text is naturally organized into hierarchical units such as paragraphs, sentences, and words. We can leverage this inherent structure to inform our splitting strategy, creating split that maintain natural language flow, maintain semantic coherence within split, and adapts to varying levels of text granularity. LangChain's RecursiveCharacterTextSplitter implements this concept:
"""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)
result= splitter.split_text(text)
print(len(result))
print(result[0])
print(result[2])
print(result[6])