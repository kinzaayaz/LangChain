from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='iris.csv')

docs = loader.load()

print(len(docs))
print(docs[4])