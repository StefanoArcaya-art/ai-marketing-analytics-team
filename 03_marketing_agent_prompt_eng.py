# PYTHON FOR GENERATIVE AI COURSE
# MULTI-AGENTS (AGENTIAL SUPERVISION)
# ***

# GOAL: Make a Marketing Email Copy Writer based on the AI Fast Track

# LIBRARIES

from customer_analytics_team.agents.marketing_agent import make_marketing_agent

from langchain_core.messages import HumanMessage

import pandas as pd

import os
import yaml

from pprint import pprint
from IPython.display import Markdown

# Backup to display mermaid graphs
from IPython.display import display, Image

# Key Inputs
MODEL = 'gpt-4o-mini'

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

# * STEP 1: MAKE THE MARKETING COPY WRITER AGENT

marketing_agent = make_marketing_agent(model=MODEL)

marketing_agent

# display(Image(marketing_agent.get_graph().draw_png()))

marketing_agent.get_input_jsonschema()['properties']


# * TEST: Write a marketing email to customers who have not purchased 'Learning Labs Pro - Paid Course'

messages = [
    HumanMessage("Write a marketing email to customers who have not purchased 'Learning Labs Pro - Paid Course'")
]
result = marketing_agent.invoke({"messages": messages})

result

Markdown(result['messages'][0].content)
