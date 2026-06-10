import os
import chromadb
import torch
from chonkie import TokenChunker
from transformers import AutoModel, AutoTokenizer
from mcp.server.fastmcp import FastMCP
from config import EMBEDDING_MODEL_PATH, KB_DATA_DIR, VECTOR_DB_DIR, MCP_HOST, MCP_PORT, TOP_K

# 1. Initialize FastMCP server and local Chroma client
mcp = FastMCP("rag-service", host=MCP_HOST, port=MCP_PORT, json_response=True, stateless_http=True)
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = chroma_client.get_or_create_collection("agent_knowledge")

# 2. Minimal model loading (removed thread limits and bfloat16 optimization)
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH, local_files_only=True)
emb_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH, local_files_only=True)
emb_model.eval()


def get_embeddings(texts):
    """Generate embeddings using a simple Mean Pooling strategy"""
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = emb_model(**inputs)
        # Simplified: Use Mean Pooling instead of complex padding-conditional logic
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().tolist()


@mcp.tool()
def retrieve_knowledge(query: str) -> str:
    """MCP Tool: Retrieve relevant context for the query"""
    formatted_query = f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}"
    results = collection.query(query_embeddings=get_embeddings([formatted_query]), n_results=TOP_K)

    docs = results.get("documents", [[]])[0]
    return "\n".join(docs) if docs else "No relevant knowledge found."


def build_knowledge_base():
    """Build the vector database from local markdown and text files"""
    if not os.path.exists(KB_DATA_DIR): return

    # Read all file contents
    all_text = ""
    for f_name in os.listdir(KB_DATA_DIR):
        if f_name.endswith((".md", ".txt")):
            with open(os.path.join(KB_DATA_DIR, f_name), "r", encoding="utf-8") as f:
                all_text += f.read() + "\n\n"

    if not all_text.strip(): return

    # Chunk text, generate embeddings, and store in Chroma
    chunks = [c.text for c in TokenChunker(tokenizer=tokenizer, chunk_size=50, chunk_overlap=10).chunk(all_text)]
    collection.add(
        documents=chunks,
        embeddings=get_embeddings(chunks),
        ids=[f"id_{i}" for i in range(len(chunks))]
    )
    print(f"Knowledge base built successfully. {len(chunks)} chunks stored.")


def clear_knowledge_base():
    """Delete the collection and recreate an empty one to clear all data"""
    global collection
    try:
        chroma_client.delete_collection("agent_knowledge")
        collection = chroma_client.get_or_create_collection("agent_knowledge")
        print("Knowledge base cleared successfully.")
    except Exception as e:
        print(f"Failed to clear knowledge base: {e}")


if __name__ == "__main__":
    # --- DB Management (Uncomment when needed, then comment out) ---
    clear_knowledge_base()
    build_knowledge_base()

    print(f"RAG MCP Server running at: http://{MCP_HOST}:{MCP_PORT}/mcp")
    mcp.run(transport="streamable-http")