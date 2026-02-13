import os
from venv import logger
from apikey import apikey


import certifi


import ssl  
import httpx
import logging

import os.path
import sys  

import contextlib

# -*- coding: utf-8 -*-
# Migrated from legacy LangChain (initialize_agent/load_tools/OpenAI LLM) to modern APIs.
# Requires:
#     pip install -U langchain langchain-openai langchain-community wikipedia numexpr streamlit
# Set env var:
#     export OPENAI_API_KEY=...  (or configure AzureOpenAI via langchain-openai docs)

import os
from typing import Optional

# --- LLM provider (modern) ---
# Old: from langchain.llms import OpenAI
# New: chat models live in provider-specific packages
from langchain_openai import ChatOpenAI

# --- Tools (modern) ---
# Prefer explicit tool imports instead of legacy load_tools
from langchain.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper


# Math tool: use the built-in LLMMathChain via community package
from langchain_classic.chains import LLMMathChain

from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent

# --- Agent building blocks ---
from langchain_classic.agents import AgentExecutor

from langchain_core.prompts import ChatPromptTemplate

import streamlit as st  

from langchain_core.tools import Tool


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

def build_agent(openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> AgentExecutor:
    """Build a modern tool-calling agent equivalent to ZERO_SHOT_REACT_DESCRIPTION.

    Args:
        openai_api_key: Optional explicit key; if None, reads from OPENAI_API_KEY.
        model: OpenAI chat model name.
    Returns:
        AgentExecutor ready to .invoke({"input": ...}).
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    # Modern chat LLM
    llm = ChatOpenAI(model=model, temperature=0.0)

    # --- Wikipedia tool (explicit) ---
    wiki = WikipediaAPIWrapper(lang="en", top_k_results=3, doc_content_chars_max=4000)

    def wiki_search(query: str) -> str:
        return wiki.run(query) # Modern .invoke with dict input  

    wiki_tool = Tool(
        name="wikipedia",
        func=wiki_search,
        description="Recherche Wikipedia. Fournir un thème clair pour obtenir un court résumé.",
    )

    # --- LLM Math tool (chain wraps the LLM for math) ---
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)

    def llm_math(q: str) -> str:
        return llm_math_chain.run(q) # Modern .invoke with dict input
    

    math_tool = Tool(

        name="llm-math",
        func=llm_math,
        description="Résout des expressions ou problèmes mathématiques en langage naturel.",
    )

    tools = [wiki_tool, math_tool]

    # --- Prompt compatible with tool-calling agent ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant utile. Utilise les outils si nécessaire pour répondre de façon factuelle et concise."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor


   
def main():

    setup_logs()

    os.environ["OPENAI_API_KEY"] = apikey      

    # ctx = ssl.create_default_context(cafile=CACERT_PATH)
    # ctx.load_cert_chain(certfile=CACERT_PATH)
    # client = httpx.Client(verify=ctx)

    st.title("Wikipedia Research Task ")
    task = st.text_input("Input Wikipedia Research Task")

    agent = build_agent()
    # Modern execution uses .invoke with a dict
    result = agent.invoke({"input": task})
    # AgentExecutor returns a dict with an 'output' field
    
    st.write(result.get("output", result)  )
    
if __name__ == "__main__":
    try:
        main()  
    except Exception as e:
        logging.error("An error occurred: %s", str(e))      

