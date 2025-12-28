from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from models.models import get_llm

llm = get_llm()


def get_rag_chain():
    """
    Creates the final RAG chain that takes {context} and {question}
    and returns the answer.
    """

    # 1. Define the Strict System Prompt
    system_prompt = """You are a Senior Legal Auditor. 
    You answer questions based strictly on the provided legal documents.
    
    CONTEXT:
    {context}
    
    RULES:
    - If the answer is not in the context, reply: "I cannot find this information in the provided documents."
    - Do NOT use outside knowledge (like generic laws) unless explicitly asked.
    - Cite the document name if available in the context metadata.
    - Format complex clauses into clear bullet points.
    """

    # 2. Build the Prompt Template
    # We use ChatPromptTemplate because DeepSeek/Llama are Chat Models
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            # This placeholder allows us to inject chat history later if needed
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", "{question}"),
        ]
    )

    # 3. Create the Chain
    # logic: (Context + Question) -> Prompt -> LLM -> String Output
    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain


def format_docs(docs):
    """
    Helper function to combine list of docs into a single string with metadata.
    This "Injects" the source name into the context so the LLM can cite it.
    """
    formatted_text = []
    for doc in docs:
        source = doc.metadata.get("source_filename", "Unknown Document")
        content = doc.page_content
        formatted_text.append(f"SOURCE: {source}\nCONTENT: {content}\n")

    return "\n---\n".join(formatted_text)
