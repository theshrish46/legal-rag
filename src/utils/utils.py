from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st


def get_chat_history():
    history = []
    for msg in st.session_state.messages[
        :-1
    ]:  # Skip the very last user msg (it's the current input)
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history
