from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_PATH = 'data'

def load_documents():
    loader = DirectoryLoader(
        path=DATA_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )

    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500, length_function=len, add_start_index=True)

    chunks = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")

    return chunks
    
split_text(load_documents())