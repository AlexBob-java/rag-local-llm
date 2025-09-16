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
