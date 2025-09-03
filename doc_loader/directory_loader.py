from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader

loader=DirectoryLoader(
    path=("books"),
    glob=("**/*.pdf"),
    loader_cls=PyPDFLoader
)

docs=loader.load()
# docs=loader.lazy_load()  for large no of docs
print(len(docs))
print(docs[2].page_content)
print(docs[2].metadata)