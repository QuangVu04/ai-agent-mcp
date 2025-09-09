from sentence_transformers import SentenceTransformer
from chromadb import chromadb
from langchain_core.tools import tool

VECTOR_DB_PATH = "./user_memory_db"
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
facts_collection = chroma_client.get_or_create_collection("user_facts")


def add_user_fact(fact: str, category: str = "general", name: str = None, fact_type: str = None):
    """Save a fact with optional metadata (name, type, category)."""
    embedding = embedder.encode(fact).tolist()
    doc_id = str(hash(fact))  # unique enough for single-user setup
    metadata = {"category": category}
    if name:
        metadata["name"] = name
    if fact_type:
        metadata["type"] = fact_type

    facts_collection.add(
        ids=[doc_id],
        metadatas=[metadata],
        documents=[fact],
        embeddings=[embedding]
    )


def query_user_facts(query: str, k=3, category: str = None, name: str = None, fact_type: str = None):
    """Query facts using both metadata filter and semantic similarity."""
    where_filter = {}
    if category:
        where_filter["category"] = category
    if name:
        where_filter["name"] = name
    if fact_type:
        where_filter["type"] = fact_type

    query_embedding = embedder.encode(query).tolist()
    results = facts_collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_filter if where_filter else None
    )

    docs = [doc for sublist in results["documents"] for doc in sublist]
    return docs


@tool
async def update_user_fact(fact: str, category: str = "general", name: str = None, fact_type: str = None) -> str:
    """
    Store a fact about the user in persistent memory.

    Args:
        fact: The content of the fact to store.
        category: The category of the fact. One of:
            - profile: Basic identity info (e.g., "Tên tôi là An", "Sinh năm 1995")
            - preference: Likes or dislikes (e.g., "Thích ăn sushi", "Không thích đồ cay")
            - goal: Important goals or plans (e.g., "Muốn thi JLPT N3 cuối năm")
            - conversation: Short-term conversational context
            - general: Miscellaneous facts the user asked to remember
        name: (optional) A subject/entity name related to the fact (e.g., "Vũ")
        fact_type: (optional) Type of fact (e.g., "email", "phone", "address")

    Returns:
        Confirmation message that the fact was saved.
    """
    add_user_fact(fact, category, name, fact_type)
    return f"Fact saved under category '{category}'."


@tool
async def query_user_fact(query: str, category: str = None, name: str = None, fact_type: str = None) -> list[str]:
    """
    Retrieve top facts about the user.

    The model should call this tool when:
    - A question refers to past user facts
    - A decision requires user's stored preferences or characteristics

    Args:
        query: Semantic query about user facts
        category: (optional) Filter by category
        name: (optional) Filter by subject/entity name
        fact_type: (optional) Filter by fact type

    Returns:
        List of up to 3 facts matching the query and filters.
    """
    return query_user_facts(query, category=category, name=name, fact_type=fact_type)
