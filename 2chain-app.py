import os
from venv import logger
from apikey import apikey

import streamlit as st  
from langchain_openai import  ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader  

from langchain_core.tracers import ConsoleCallbackHandler  


import certifi


import ssl
import httpx
import logging

import os.path
import sys  

import contextlib  


CACERT_PATH = "cacert.pem"

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

def check_cacerts():

    logger.info(f"Chemin par default pour les certificats : {certifi.where()  }")

    defaultPath = os.getcwd()

    logging.info("Current working directory: %s", defaultPath  )

    if not os.path.isfile(CACERT_PATH):
        logging.info(f"{CACERT_PATH} not found !!!! "    )
    else:
        logging.info(f"{CACERT_PATH} found !!!! "    )    

   
def main():

    setup_logs()

    os.environ["OPENAI_API_KEY"] = apikey      

    # ctx = ssl.create_default_context(cafile=CACERT_PATH)
    # ctx.load_cert_chain(certfile=CACERT_PATH)
    # client = httpx.Client(verify=ctx)

    st.title("Medium Article Generator ")
    topic = st.text_input("Enter the topic for the article:")
    language = st.text_input("Enter the language for the article:")

    title_template = ChatPromptTemplate(
        [
            ("system", "You are a helpful assistant that generates title content based on a given topic."),
            ("human", "Generate an article  for the following topic: {topic} in {language}" )
        ]
    )

    article_template = ChatPromptTemplate(
        [
            ("system", "You are a helpful assistant that generates full article content based on a given topic."),
            ("human", "Generate an article  for the following topic: {topic} in {language}" )
        ]
    )


    llm = ChatOpenAI(model="gpt-4o", temperature=0.9 )

    llm2 = ChatOpenAI(model="gpt-4o-mini", temperature=0.9 )

    titlechain = title_template |  llm  | StrOutputParser() 

    articlechain = title_template |  llm2  | StrOutputParser() 


    if topic:
        response1 = titlechain.invoke( {"topic": topic, "language": language},config={'callbacks': [ConsoleCallbackHandler()]})
        response = articlechain.invoke( {"topic": topic, "language": language},config={'callbacks': [ConsoleCallbackHandler()]})    
        st.write(response1 + "\n\n" + response  )

try:
    main()  
except Exception as e:
    logging.error("An error occurred: %s", str(e))      

