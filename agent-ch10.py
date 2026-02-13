import os
from apikey import apikey
import streamlit as st 
import logging 
import sys 


# --- LLM provider (modern) ---
# Old: from langchain.llms import OpenAI
# New: chat models live in provider-specific packages
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough



def setup_logs():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(filename)s.%(lineno)d:%(funcName)s - %(asctime)s - %(levelname)s - %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler('langchain-crash-course.log' , mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)


def format_docs(docs):
    """Utility: fold retrieved Documents into a single context string."""
    return "".join(d.page_content for d in docs)


def build_rag_chain(vectorstore, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    """
    Build a Retrieval-Augmented Generation chain using LCEL (LangChain Expression Language).
    Replaces the legacy `RetrievalQA` chain.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant utile et factuel. Utilise strictement le contexte fourni pour répondre. Si l'information n'est pas dans le contexte, réponds que tu ne sais pas."),
        ("human", "Question:{question} Contexte:{context}")
    ])

    model = ChatOpenAI(model=model_name, temperature=temperature)
    parser = StrOutputParser()

    # The input to the chain will be the user's question (a string).
    # It is passed to both the retriever (to get {context}) and to the prompt as {question}.
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | parser
    )
    return rag_chain


def main():
    st.title("LangChain Crash Course – RAG (LCEL) Example")

    os.environ["OPENAI_API_KEY"] = apikey    

    # --- LLM & embeddings setup ---
    # NOTE: gpt-3.5-turbo is deprecated; use a current small model like gpt-4o-mini (or your Azure OpenAI deployment).
    model_name = st.sidebar.text_input("OpenAI chat model", value="gpt-4o-mini")
    temperature = st.sidebar.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    # --- Load and split documents ---
    source_path = st.sidebar.text_input("Chemin du texte source", value="constitution_fr.txt")
    if not os.path.exists(source_path):
        st.info(f"Le fichier '{source_path}' n'existe pas encore. Placez-le à la racine ou changez le chemin dans la barre latérale.")
        # We still allow typing the question, but retrieval will fail until file exists

    if os.path.exists(source_path):
        loader = TextLoader(source_path, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # --- Create embeddings and vector store ---
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma.from_documents(docs, embeddings)

        # --- Build retrieval-augmented chain (LCEL) ---
        rag_chain = build_rag_chain(vectorstore, model_name=model_name, temperature=temperature)
    else:
        rag_chain = None

    # --- Streamlit UI ---
    query = st.text_input("Posez votre question sur le document :")

    if query:
        if rag_chain is None:
            st.error("Impossible de répondre : aucun corpus n'est indexé (fichier source introuvable).")
        else:
            with st.spinner("Génération de la réponse..."):
                try:
                    answer = rag_chain.invoke(query)
                    st.write("**Réponse :**")
                    st.write(answer)
                except Exception as e:
                    st.exception(e)


if __name__ == "__main__":
    setup_logs()
    main()      

