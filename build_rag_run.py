# build_rag.py
import os
import glob
import json
import json5
import torch
from tqdm import tqdm
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Parameters
CODEBASE_DIR = "./codebase_small"
OUTPUT_DIR = "./faiss_index_semantic"
CHUNKS_DIR = "./chunks_semantic"
EMBEDDINGS_MODEL_NAME = "BAAI/bge-large-en-v1.5"
LLM_MODEL_NAME = "deepseek-coder:6.7b"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDINGS_MODEL_NAME,
    model_kwargs={"device": device},
    encode_kwargs={"device": device, "batch_size": 32},
)

# Initialize LLM
llm = OllamaLLM(
    model=LLM_MODEL_NAME,
    temperature=0.0,
    num_predict=1024,
)

# ================== HELPERS ==================
def get_code_files():
    java_files = glob.glob(os.path.join(CODEBASE_DIR, "**/*.java"), recursive=True)
    kt_files = glob.glob(os.path.join(CODEBASE_DIR, "**/*.kt"), recursive=True)
    return java_files + kt_files

def make_chunk_path(file_path: str) -> str:
    return os.path.join(CHUNKS_DIR, os.path.basename(file_path) + ".json")

def make_fullclass_chunk_path(file_path: str) -> str:
    return os.path.join(CHUNKS_DIR, os.path.basename(file_path) + ".full.json")

def save_json_or_txt(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        txt_path = path.replace(".json", ".txt")
        print(f"⚠️ Failed JSON save for {path}, fallback to TXT. Error: {e}")
        with open(txt_path, "w", encoding="utf-8") as f:
            if isinstance(data, list):
                for d in data:
                    if isinstance(d, dict):
                        f.write(d.get("content", "") + "\n\n")
                    else:
                        f.write(str(d) + "\n")
            elif isinstance(data, dict):
                f.write(data.get("content", ""))
            else:
                f.write(str(data))

def load_json_if_exist(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to parse {path}, skipping. Error: {e}")
    return None

def sanitize_chunk(chunk: dict) -> dict:
    sanitized = chunk.copy()
    sanitized["file_path"] = sanitized.get("file_path", "").replace("\\", "/")
    content = sanitized.get("content", "")
    content = content.replace("(...)", "")
    content = content.replace("$", "\\$")
    sanitized["content"] = content
    return sanitized

# ================== SEMANTIC SPLIT ==================
def semantic_split(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    prompt = f"""
You are a code analyst. Split the following Java/Kotlin file into meaningful semantic parts.
For each part, return:
- name: function or class name or endpoint
- type: one of function, class, endpoint, helper, model, config, test
- content: code snippet

Respond in strict JSON array format ONLY. No explanations, no comments.

File content:
{content}
"""
    chunks = []
    try:
        response = llm(prompt)
        try:
            chunks = json.loads(response)
        except Exception:
            chunks = json5.loads(response)
    except Exception as e:
        print(f"⚠️ Failed to parse JSON for {file_path}, using plain text. Error: {e}")
        # fallback plain-text chunk
        chunks = [{
            "name": os.path.splitext(os.path.basename(file_path))[0],
            "type": "plain_text",
            "content": content,
            "file_path": file_path.replace("\\", "/")
        }]

    # Sanitize each chunk
    return [sanitize_chunk(c) for c in chunks]

# ================== FULL CLASS ==================
def extract_full_class_chunk(file_path: str):
    json_path = make_fullclass_chunk_path(file_path)
    existing = load_json_if_exist(json_path)
    if existing:
        return [existing]

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    full_class_chunk = {
        "name": os.path.splitext(os.path.basename(file_path))[0],
        "type": "class_full",
        "content": content,
        "file_path": file_path.replace("\\", "/")
    }

    full_class_chunk = sanitize_chunk(full_class_chunk)
    save_json_or_txt(json_path, full_class_chunk)
    return [full_class_chunk]

# ================== BUILD DOCS ==================
def build_documents():
    documents = []
    files = get_code_files()
    print(f"Found {len(files)} files.")

    for file_path in tqdm(files, desc="Processing files"):
        # Semantic chunks
        sem_path = make_chunk_path(file_path)
        sem_chunks = load_json_if_exist(sem_path)
        if sem_chunks is None:
            sem_chunks = semantic_split(file_path)
            save_json_or_txt(sem_path, sem_chunks)

        # Full class chunks
        full_chunks = extract_full_class_chunk(file_path)

        # Merge
        all_chunks = (sem_chunks or []) + full_chunks

        for chunk in all_chunks:
            doc = Document(
                page_content=chunk["content"],
                metadata={
                    "name": chunk.get("name", ""),
                    "type": chunk.get("type", ""),
                    "source": chunk.get("file_path", ""),
                },
            )
            documents.append(doc)

    print(f"Created {len(documents)} documents for FAISS.")
    return documents

# ================== FAISS ==================
def build_faiss_index(documents):
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vectorstore.save_local(OUTPUT_DIR)
    print(f"✅ RAG semantic index saved to {OUTPUT_DIR}/")
    return vectorstore

if __name__ == "__main__":
    docs = build_documents()
    if docs:
        build_faiss_index(docs)
