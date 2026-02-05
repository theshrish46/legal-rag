from langchain_core.messages import HumanMessage, AIMessage


def convert_to_langchain_messages(messages):
    """
    Converts Streamlit-style dict history to LangChain Message objects.
    Expects messages to be a list of: {"role": "user/assistant", "content": "..."}
    """
    lc_messages = []
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages
