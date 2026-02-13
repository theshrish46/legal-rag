from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import StrOutputParser


from ..llm_model.llm import get_gemini_llm
from ..database import get_vector_store
from src.retriever import get_legal_retriever


class LegalAuditResponse(BaseModel):
    answer: str = Field(description="The summary of the legal finings")
    citations: List[str] = Field(
        description="List of filenames used to support the answer"
    )
    risk_level: str = Field(description="Low, Medium, or High based on the content")


def format_docs(docs):
    """Refined helper to inject source metadata clearly."""
    return "\n\n".join(
        f"--- DOCUMENT: {d.metadata.get('source_filename', 'N/A')} ---\n{d.page_content}"
        for d in docs
    )


def get_rag_chain():
    llm = get_gemini_llm()
    retriever = get_legal_retriever()

    system_prompt = """You are a Senior Legal Auditor. Answer strictly based on context.
    
    If the answer isn't present, say: "I apologize, but the provided documents do not contain information regarding [topic]."
    
    CONTEXT:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    setup_and_retrieval = RunnableParallel(
        {
            "context": (lambda x: x["question"])
            | retriever,  # Keep as Doc objects here
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
    )

    # We use a chain that preserves context
    rag_chain = setup_and_retrieval | {
        "answer": (
            lambda x: {
                "context": format_docs(x["context"]),
                "question": x["question"],
                "chat_history": x["chat_history"],
            }
        )
        | prompt
        | llm
        | StrOutputParser(),
        "retrieved_docs": lambda x: x["context"],  # Pass the docs through for audit
    }

    return rag_chain
