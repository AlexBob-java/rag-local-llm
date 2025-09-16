# 🧑‍💻 From Corporate Codebases to My Pet Project: Building RAG for Code with LLMs  

## 🔎 Background  
When working with large codebases, I noticed a big limitation of LLMs (ChatGPT, Claude, LLaMA, etc.):  
They can explain **general programming concepts**, but they **don’t know your specific codebase**.  

That means questions like:  
- *“Where do we handle checkout logic?”*  
- *“How do we validate user payments?”*  
- *“Can you write a unit test for the CartService?”*  

…were impossible to answer out-of-the-box.  

So I decided to create a **pet project**: a Retrieval-Augmented Generation (RAG) system over my own codebase.

---

## 🛠️ The Idea  
The project had 3 goals:  
1. **Index my codebase** (Java, Kotlin, YAML).  
2. **Ask natural-language questions** about the code.  
3. **Generate unit/integration tests** based on retrieved snippets.  

---

## ⚙️ Tech Stack  
- **Vector Store**: FAISS (fast semantic search).  
- **Embeddings**: `BAAI/bge-large-en-v1.5` (HuggingFace).  
- **LLM**: Ollama with `llama3.1:8b` (runs locally).  
- **Framework**: LangChain.  
- **UI**: Gradio web app (chat interface).
- **OS** Win11
- **Hardware** Laptop with Nvidia RTX 2060 6Gb, AMD Ryzen 7 4800HS, RAM 24Gb.

---

## 🧩 How It Works  
1. **Chunking**  
   - Split the codebase into small chunks.  
   - Try JSON format first.  
   - If JSON is invalid → fallback to plain text.  
   - Both are stored in FAISS.  

2. **Embedding & Indexing**  
   - Each chunk is embedded with `bge-large-en`.  
   - Stored in FAISS for fast retrieval.  

3. **Query Flow**  
   - User asks a question in the chat.  
   - Retriever pulls top-k relevant code snippets.  
   - LLM combines them with a custom prompt → produces an answer.  

4. **UI**  
   - Clean chat interface with history.  
   - Auto-scroll fixed by storing pairs `[user, assistant]`.  
   - Button to clear chat history.  

---

## 💡 Example Queries  

```text
Q: Where is payment validation implemented?  
A: In PaymentValidator.java, inside the validatePayment() method.  
   It checks for null card data, expiration date, and compares the amount 
   with the user’s order total.

Q: Can you generate a unit test for CartService?  
A: Sure. Here’s a JUnit5 test using Mockito:

@Test
void testAddItemToCart() {
    CartService cartService = new CartService();
    cartService.addItem("user123", "item456");

    assertEquals(1, cartService.getItems("user123").size());
}
```
# 🔧 Prerequisites & Setup Guide

This project demonstrates a **RAG (Retrieval-Augmented Generation)** pipeline with Ollama, FAISS, LangChain, and Gradio for interactive codebase Q&A.

---

## 1. Install Python and Virtual Environment
1. Make sure you have **Python 3.10+** installed:
   ```bash
   python3 --version
2. Create a virtual environment:
   ```bash
   python3 -m venv rag_env
   source rag_env/bin/activate   # Linux / macOS
   rag_env\Scripts\activate      # Windows

## 2. Install Ollama
1. Download and install Ollama: https://ollama.ai

macOS:
   ```bash
   brew install ollama
   ```
Linux: follow instructions from the website <br>
Windows: download and install the .msi package
Start the Ollama service:
   ```bash
   ollama serve
   ```

Pull the required models:
   ```bash
   ollama pull deepseek-coder:6.7b
   ollama pull llama3.1:8b
   ```
## 3. Install Python Dependencies
Inside the virtual environment, install the required packages:
   ```bash
   pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
   pip install gradio
   pip install langchain langchain-community langchain-huggingface langchain-ollama
   pip install faiss-cpu
   pip install json5 tqdm
   ```
👉 If you have a GPU, install the CUDA version of PyTorch instead:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```
## 4. Prepare Your Codebase
Place your Java/Kotlin files into a folder, for example:
   ```bash
   ./codebase_small/
   ```
This folder will be processed by build_rag.py to create semantic chunks.

## 5. Build the RAG Index
Run the index builder script:
   ```bash
   python build_rag_run.py
   ```
This will:
* Split your code into semantic chunks (JSON or plain text)
* Save them in ./chunks_semantic/
* Build a FAISS vector index in ./faiss_index_semantic/
## 6. Launch the Web App
Run the Gradio chatbot interface:
   ```bash
   python web_rag_run.py
   ```
Then open http://localhost:7860
 in your browser and start asking questions about your codebase 🎉
