

from langchain_core.messages import HumanMessage

import os
import yaml

from IPython.display import Markdown

# Key Inputs
MODEL = 'gpt-4o-mini'
EMBEDDING = 'text-embedding-ada-002'
PATH_VECTORDB = "data/data-rag-product-information/products_clean.db"

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

# * STEP 1: CREATE THE VECTOR DATABASE


# * STEP 2: MAKE THE AGENT


# * STEP 3: MODULARIZE THE AGENT

from customer_analytics_team.agents.product_expert import make_product_expert_agent

# Make the agent

product_expert_agent = make_product_expert_agent(
    model=MODEL,
    model_embedding=EMBEDDING,
    db_path=PATH_VECTORDB
)

product_expert_agent

product_expert_agent.get_input_jsonschema()



# TEST: WHAT IS LEARNING LABS PRO?

messages = [
    HumanMessage("What is Learning Labs PRO?")
]

result = product_expert_agent.invoke({"messages": messages})

Markdown(result['messages'][0].content)
