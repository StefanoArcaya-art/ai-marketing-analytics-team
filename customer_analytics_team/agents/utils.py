from langchain_core.messages import HumanMessage, AIMessage

# Helper functions to get last question that the Human asked
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
