# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# MULTI-AGENTS (AGENTIAL SUPERVISION)
# ***

# GOAL: Make a product expert AI agent based on the RAG agent from Clinic #1

# LIBRARIES

# Agent Messages
from langchain_core.messages import HumanMessage

# Modular Agent 
from marketing_analytics_team.agents.product_expert import make_product_expert_agent

# Environment
import os
import yaml
from pprint import pprint
from IPython.display import Markdown
from IPython.display import display, Image

# Key Inputs
MODEL = 'gpt-4.1-mini'
EMBEDDING = 'text-embedding-ada-002'
PATH_VECTORDB = "data/data-rag-product-information/products_clean.db"

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('credentials.yml'))['openai']


# * STEP 1: MAKE THE RAG AGENT
#   * NEW: Implement AIMessage responses to return AI reasoning (refer to marketing_analytics_team.agents.product_expert)

# Make the agent

product_expert_agent = make_product_expert_agent(
    model=MODEL,
    model_embedding=EMBEDDING,
    db_path=PATH_VECTORDB
)

product_expert_agent

# display(Image(product_expert_agent.get_graph().draw_png()))

# Get the input schema
product_expert_agent.get_input_jsonschema()['properties']



# * TEST: What is Learning Labs PRO?

messages = [
    HumanMessage("What is Learning Labs PRO? Include a summary of the course, how many labs, and costs.")
]

result = product_expert_agent.invoke({"messages": messages})

result.keys()

result['response']

Markdown(result['response'][0].content)

# pprint(result['response'][0].content)


# * TEST: How long will it take to complete the 5-Course R-Track?

messages = [
    HumanMessage("Estimate how long will it take to complete the 5-Course R-Track.")
]

result = product_expert_agent.invoke({"messages": messages})

Markdown(result['response'][0].content)

# pprint(result['response'][0].content)

# * TEST: How long will it take to complete the 5-Course R-Track?

messages = [
    HumanMessage("Collect information on the 5-Course R Track")
]

result = product_expert_agent.invoke({"messages": messages})

Markdown(result['response'][0].content)

# * CONCLUSIONS
#   - This is the Product Expert Agent based on RAG architecture from Clinic #1
#   - The only difference is now we are using AIMessage to get the reasoning steps and the final answer
#   - We will use this later to create a multi-agent system for marketing analytics
