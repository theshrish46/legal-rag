from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from models_utils.models import get_llm

llm = get_llm()


def get_rag_chain():
    system_prompt = """You are a Senior Legal Auditor. 
    You answer questions based strictly on the provided legal documents.
    
    CONTEXT:
    {context}
    
    RULES:
    - If the answer is not in the context, reply: like a human with politensess."
    - Do NOT use outside knowledge (like generic laws) unless explicitly asked.
    - Cite the document name if available in the context metadata.
    - Format complex clauses into clear bullet points.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
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
