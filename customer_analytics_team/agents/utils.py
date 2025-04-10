from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import BaseOutputParser

import re


# * Helper functions to get last question that the Human asked
def get_last_human_message(msgs):
    # Iterate through the list in reverse order
    for msg in reversed(msgs):
        if isinstance(msg, HumanMessage):
            return msg
    return None

def get_last_ai_message(msgs, target_name=None):
    for msg in reversed(msgs):
        if not target_name:
            if isinstance(msg, AIMessage):
                return msg
        if target_name:
            if isinstance(msg, AIMessage) and msg.name == target_name:
                return msg
    return None

# * Output Parsers for SQL and Python (Business Intelligence Agent)

def extract_sql_code(text):
    sql_code_match = re.search(r'```sql(.*?)```', text, re.DOTALL)
    sql_code_match_2 = re.search(r"SQLQuery:\s*(.*)", text)
    if sql_code_match:
        sql_code = sql_code_match.group(1).strip()
        return sql_code
    if sql_code_match_2:
        sql_code = sql_code_match_2.group(1).strip()
        return sql_code
    else:
        sql_code_match = re.search(r"sql(.*?)'", text, re.DOTALL)
        if sql_code_match:
            sql_code = sql_code_match.group(1).strip()
            return sql_code
        else:
            return None
        
def extract_python_code(text):
    python_code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
    if python_code_match:
        python_code = python_code_match.group(1).strip()
        return python_code
    else:
        python_code_match = re.search(r"python(.*?)'", text, re.DOTALL)
        if python_code_match:
            python_code = python_code_match.group(1).strip()
            return python_code
        else:
            return None

class SQLOutputParser(BaseOutputParser):
    def parse(self, text: str):
        sql_code = extract_sql_code(text)
        if sql_code is not None:
            return sql_code
        else:
            # Assume ```sql wasn't used
            return text

class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):        
        python_code = extract_python_code(text)
        if python_code is not None:
            return python_code
        else:
            # Assume ```python wasn't used
            return text

