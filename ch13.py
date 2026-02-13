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
    logger.setLevel(logging.DEBUG)
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


def build_rag_chain(vectorstore, model_name: str = "gpt-5.1", temperature: float = 0.0):
    """ 
    Build a Retrieval-Augmented Generation chain using LCEL (LangChain Expression Language).
    Replaces the legacy `RetrievalQA` chain.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
 
    question = st.session_state.query 
    
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

def clear_history():
    if "history" in st.session_state:
        st.session_state['history']  = []    

def process_uploaded_file(uploaded_file):

    if uploaded_file is None:
        st.error("Aucun fichier uploadé. Veuillez uploader un fichier pour continuer.")
        return None 

    with st.spinner(f"Loading {uploaded_file.name} ..."):

        bytes_data = uploaded_file.read()
        fileName = os.path.join('.' , 'TEMP', uploaded_file.name)
        with open(fileName, "wb") as f:
            f.write(bytes_data) 

    with st.spinner(f"Processing {uploaded_file.name} ..."):
        loader = TextLoader(fileName, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # --- Create embeddings and vector store ---
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma.from_documents(docs, embeddings)

    st.success(f"File '{uploaded_file.name}' uploaded and processed successfully.") 

    return vectorstore      


def on_add_file():
    clear_history() 
    
    vectorstore = process_uploaded_file( st.session_state.uploaded_file  )

    if vectorstore is not None:

        st.session_state.rag = build_rag_chain(vectorstore=vectorstore, 
                                               model_name=st.session_state.model_name, 
                                               temperature=st.session_state.temperature) 
        
        logging.info("RAG chain initialized with uploaded file.")


def init_session():
    logging.info("Initializing session state.") 
    if "rag" not in st.session_state:
        st.session_state.rag = None
        st.session_state.history = []
        st.session_state.model_name = "gpt-5.1"
        st.session_state.temperature = 0.0  
        st.session_state.uploaded_file = None 
        st.session_state.query = None

def main():

    logging.info("Starting main function.")

    init_session()  

    st.title("Chat with document")

    os.environ["OPENAI_API_KEY"] = apikey    

    st.session_state.uploaded_file = st.file_uploader("Upload a text file", type=["txt", "pdf", "docx" ])

    add_file = st.button("Add file " , on_click=on_add_file)    
    # --- LLM & embeddings setup ---
    # NOTE: gpt-3.5-turbo is deprecated; use a current small model like gpt-4o-mini (or your Azure OpenAI deployment).
    st.session_state.model_name = st.sidebar.text_input("OpenAI chat model", value="gpt-5.1")
    st.session_state.temperature = st.sidebar.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)



        # --- Streamlit UI ---
    st.session_state.query = st.text_input("Posez votre question sur le document :")

    if st.session_state.query:

        if st.session_state.rag is None:
            st.error("Impossible de répondre : aucun corpus n'est indexé. Veuillez uploader un fichier et cliquer sur 'Add file' pour initialiser la chaîne RAG.")
        else:
            with st.spinner("Génération de la réponse..."):
                try:
                    answer = st.session_state.rag.invoke(st.session_state.query) 
                    st.write("**Réponse :**")
                    st.write(answer)
                    st.session_state.history.append(st.session_state.query)        
                    # Display the conversation history
                    for line in st.session_state.history: 
                        st.sidebar.write(line)
                except Exception as e:
                    st.exception(e)


if __name__ == "__main__":
    setup_logs()
    main()      

