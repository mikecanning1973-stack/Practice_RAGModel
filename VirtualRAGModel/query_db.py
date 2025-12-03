import os
from dotenv import load_dotenv
import argparse
from get_embedding_func import get_embedding_func
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM

load_dotenv('configuration.env')
COLLECTION_PATH = os.getenv('COLLECTION_PATH')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
PDF_PATH = os.getenv('PDF_PATH')
LLM_QUERY_MODEL = os.getenv('LLM_QUERY_MODEL')
CHUNKS_TO_RETURN= int(os.getenv('CHUNKS_TO_RETURN'))
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE'))

#PROMPT_TEMPLATE = """
#Answer the question based only on the following context if no answer is found respond with "unsure":

#{context}

#---

#Answer the question based on the above context: {question}
#"""

PROMPT_TEMPLATE = """
You are a chatbot named Aero Space The users of this are Systems engineers that are using this for researching Systems Engineering 
principles and best practices if no answer is found respond with "I was not able to find the answer based on the provided context provided".:

{context}

---

Answer the questions as systems engineer would based on the above context: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_func()
    db = Chroma(persist_directory=COLLECTION_PATH, collection_name=COLLECTION_NAME, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=CHUNKS_TO_RETURN)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    #print("\033[92m" + prompt + "\033[0m")

    #model = Ollama(model=LLM_QUERY_MODEL)
    model = OllamaLLM(model=LLM_QUERY_MODEL, temperature=LLM_TEMPERATURE)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
#   formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(f"\n\033[92mResponse: {response_text}\033[0m")
    print(f"\n\033[93mSources: {sources}\033[0m")
    return response_text

if __name__ == "__main__":
    main()