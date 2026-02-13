from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import retrieval_qa
import os

from src.database import get_vector_store

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", api_key=os.environ.get("GOOGLE_API_KEY")
)
vectorstore = get_vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def rag_pipeline(input):
    # Extract retrieval context
    retrieved_docs = retriever.get_relevant_documents(input)
    context_texts = [doc.page_content for doc in retrieved_docs]

    # Generate response
    qa_chain = retrieval_qa.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    result = qa_chain.invoke({"query": input})
    return result["result"], context_texts
