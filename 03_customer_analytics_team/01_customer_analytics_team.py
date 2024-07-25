

# * LIBRARIES

from langchain_community.vectorstores import Chroma

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import pandas as pd
import yaml
import os

from pprint import pprint

# * DATABASE SETUP

PATH_PRODUCTS_VECTORDB = "data/data-rag-product-information/products_clean.db"


# * AI SETUP

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

OPENAI_LLM = ChatOpenAI(
    model = "gpt-4o-mini"
)

# *** PRODUCTS RAG AGENT ****

def create_rag_agent(db_path, llm, temperature = 0):
    
    embedding_function = OpenAIEmbeddings(
        model='text-embedding-ada-002',
    )
    
    model = llm
    
    model.temperature = temperature
    
    vectorstore_2 = Chroma(
        embedding_function=embedding_function, 
        persist_directory=db_path
    )

    retriever_2 = vectorstore_2.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:
            {context}

            Question: {question}
        """
    )

    rag_chain = (
        {"context": retriever_2, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain


products_expert_agent = create_rag_agent(PATH_PRODUCTS_VECTORDB, llm=OPENAI_LLM, temperature=0.7)

# result = products_expert_agent.invoke("Is the 4-Course R-Track Open for Enrollment?")
# result


