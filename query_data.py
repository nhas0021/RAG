import argparse
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load API key from .env
load_dotenv()

# Path to existing Chroma DB
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are an analytical assistant.

Answer the question using ONLY the information provided in the context below.
Do NOT use outside knowledge.
If the context does not contain enough information to answer the question, say:
"I don't have enough information in the provided context to answer this question."

Context:
{context}

---

Question:
{query}

Answer:
"""

def main():
    #Read query from CLI
    parser = argparse.ArgumentParser(description="Query the Chroma database")
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    query_text = args.query_text

    #Load the existing DB
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
    )

    #Search the DB
    results = db.similarity_search(query_text, k=3)

    #Combine retrieved chunks
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

    #create prompt to be sent to LLM (The context text)
    prompt = PROMPT_TEMPLATE.format(
    context=context_text,
    query=query_text)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = model.invoke(prompt)
    response_text = response.content

    #Print final answer
    formatted_response = (
        f"\n=== ANSWER ===\n{response_text}\n\n"
    )

    print(formatted_response)



if __name__ == "__main__":
    main()
