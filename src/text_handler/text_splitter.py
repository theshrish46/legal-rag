from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_recursive_text_splitter(chunk_size=500, chunk_overlap=100):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
