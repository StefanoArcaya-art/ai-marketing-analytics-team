# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# MULTI-AGENTS (AGENTIAL SUPERVISION)
# ***

# GOAL: Make a product expert AI agent based on the RAG agent from Clinic #1

# LIBRARIES

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Rag Agents
from langchain_chroma import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# Other Libraries
import pandas as pd
import joblib
import re

import os
import yaml

from pprint import pprint
from IPython.display import Markdown

# Backup to display mermaid graphs
from IPython.display import display, Image

# Key Inputs
MODEL = 'gpt-4.1-nano'
EMBEDDING = 'text-embedding-ada-002'
PATH_VECTORDB = "data/data-rag-product-information/products_clean.db"

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

# * STEP 1: CREATE THE VECTOR DATABASE

# * Test out loading a single webpage
#   Resource: https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html

url = "https://university.business-science.io/p/4-course-bundle-machine-learning-and-web-applications-r-track-101-102-201-202a"

# Create a document loader for the website
loader = WebBaseLoader(url)

# Load the data from the website
documents = loader.load()

pprint(documents[0].metadata)

dict(documents[0]).keys()

pprint(documents[0].page_content)


# * Load All Webpages
#   This will take a minute

df = pd.read_csv("data/data-rag-product-information/products.csv")

df['website']

loader = WebBaseLoader(df['website'].tolist())

documents = loader.load()

documents[1].metadata

len(documents[1].page_content)

# joblib.dump(documents, "data/data-rag-product-information/products.pkl")

documents = joblib.load("data/data-rag-product-information/products.pkl")


# * Clean the Beautiful Soup Page Content

def clean_text(text):

    text = re.sub(r'\n+', '\n', text) 
    text = re.sub(r'\s+', ' ', text)  

    text = re.sub(r'Toggle navigation.*?Business Science', '', text, flags=re.DOTALL)
    text = re.sub(r'© Business Science University.*', '', text, flags=re.DOTALL)

    # Replace encoded characters
    text = text.replace('\xa0', ' ')
    text = text.replace('ðŸŽ‰', '')  

    # Extract relevant content
    relevant_content = []
    lines = text.split('\n')
    for line in lines:
        if any(keyword in line for keyword in ["Enroll in Course", "data scientist", "promotion", "salary", "testimonial"]):
            relevant_content.append(line.strip())

    # Join the relevant content back into a single string
    cleaned_text = '\n'.join(relevant_content)

    return cleaned_text

# Test cleaning a single document

pprint(documents[1].page_content)

pprint(clean_text(documents[1].page_content))

pprint(clean_text(documents[0].page_content))


# Clean all documents

documents_clean = documents.copy()

for document in documents_clean:
    document.page_content = clean_text(document.page_content)
    
documents_clean

len(documents_clean)

# Assess Length

for document in documents_clean:
    print(document.metadata)
    print(len(document.page_content))
    print("---")

# * Chunk into 500-1000 using Recursive Splitter

CHUNK_SIZE = 1000

text_splitter_recursive = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE,
    chunk_overlap=100,
)

documents_clean_recursive = text_splitter_recursive.split_documents(documents_clean)

documents_clean_recursive

len(documents_clean_recursive)

# * Text Embeddings
# OpenAI Embeddings
# - See Account Limits for models: https://platform.openai.com/account/limits
# - See billing to add to your credit balance: https://platform.openai.com/account/billing/overview

embedding_function = OpenAIEmbeddings(
    model=EMBEDDING,
)

# ** Vector Store - Recursively Split Documents

# Create the Vector Store (Run 1st Time)
# vectorstore_1 = Chroma.from_documents(
#     documents_clean_recursive, 
#     embedding=embedding_function, 
#     persist_directory="data/data-rag-product-information/products_recursive_3.db"
# )

# Connect to the Vector Store (Run all other times)
vectorstore_1 = Chroma(
    embedding_function=embedding_function, 
    persist_directory="data/data-rag-product-information/products_recursive_2.db"
)

vectorstore_1

vectorstore_1.similarity_search("Is the 4-Course R-Track Open for Enrollment?", k = 4)


retriever_1 = vectorstore_1.as_retriever()

retriever_1

# ** Vector Store - Complete (Large) Documents

# Create the Vector Store (Run 1st Time)
# vectorstore_2 = Chroma.from_documents(
#     documents_clean, 
#     embedding=embedding_function, 
#     persist_directory="data/data-rag-product-information/products_clean_2.db"
# )

# Connect to the Vector Store (Run all other times)
vectorstore_2 = Chroma(
    embedding_function=embedding_function, 
    persist_directory="data/data-rag-product-information/products_clean_2.db"
)

vectorstore_2

vectorstore_2.similarity_search("Is the 4-Course R-Track Open for Enrollment?", k = 4)

retriever_2 = vectorstore_2.as_retriever()

# * Prompt template 

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# * LLM Specification

model = ChatOpenAI(
    model = MODEL,
    temperature = 0.7,
)

response = model.invoke("What is the 4-Course R-Track?")

Markdown(response.content)

# * RAG Chain

# * Test 1: With Recursive Chunking

rag_chain_1 = (
    {"context": retriever_1, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

result = rag_chain_1.invoke("Is the 4-Course R-Track Open for Enrollment?")

Markdown(result)

# * Test 2: With No Chunking

rag_chain_2 = (
    {"context": retriever_2, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

result = rag_chain_2.invoke("Is the 4-Course R-Track Open for Enrollment?")

Markdown(result)

# * Conclusion: 
# - The Non-Chunked LLM quickly discovered that the 4-course R-Track was closed for enrollment


# * STEP 2: MAKE THE RAG AGENT
#  - Create a RAG Agent based on the one used in Clinic #1
#  - Modularize the agent for easier re-use in production
#  - Use LangGraph to manage State
#  - Implement LangGraph Messages History to track multi-agent conversations

# Libraries 
from customer_analytics_team.agents.product_expert import make_product_expert_agent

# Make the agent

product_expert_agent = make_product_expert_agent(
    model=MODEL,
    model_embedding=EMBEDDING,
    db_path=PATH_VECTORDB
)

product_expert_agent

# display(Image(product_expert_agent.get_graph().draw_png()))

product_expert_agent.get_input_jsonschema()



# * TEST: What is Learning Labs PRO?

messages = [
    HumanMessage("What is Learning Labs PRO? Include a summary of the course, how many labs, and costs.")
]

result = product_expert_agent.invoke({"messages": messages})

pprint(result['response'])

Markdown(result['response'][0].content)


# * TEST: How long will it take to complete the 5-Course R-Track?

messages = [
    HumanMessage("Estimate how long will it take to complete the 5-Course R-Track.")
]

result = product_expert_agent.invoke({"messages": messages})

Markdown(result['messages'][0].content)


# * TEST: How long will it take to complete the 5-Course R-Track?

messages = [
    HumanMessage("Collect information on the 5-Course R Track")
]

result = product_expert_agent.invoke({"messages": messages})

Markdown(result['response'][0].content)
