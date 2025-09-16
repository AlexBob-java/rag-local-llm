# web_rag2.py
# Веб-интерфейс для RAG с FAISS и Ollama
# Запуск: python web_rag2.py

import os
import torch
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ==========================
# Конфигурация
# ==========================
INDEX_DIR = "./faiss_index_semantic"
DEFAULT_LLM = "llama3.1:8b"
TOP_K = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ==========================
# Загрузка эмбеддера
# ==========================
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# ==========================
# Загрузка индекса FAISS
# ==========================
def load_rag_index():
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(f"Error: {INDEX_DIR} not found. Run build_rag2.py first.")
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"Loaded RAG index from {INDEX_DIR}.")
    return vectorstore

vectorstore = load_rag_index()

# ==========================
# Создание RAG-чейна
# ==========================
def create_rag_chain(model_name: str):
    llm = OllamaLLM(
        model=model_name,
        temperature=0.1,
        num_predict=512
    )

    prompt_template = """You are an expert in Java and Kotlin code analysis.
Use the following retrieved code snippets from the codebase to answer the user's question.
Provide helpful advice, explanations, or suggestions based on the code.
If the snippets are not relevant, say so and reason step-by-step.

Retrieved context:
{context}

Question: {question}

Answer (concise, code-focused):"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return qa_chain

# ==========================
# Обработка сообщений
# ==========================
def chat_fn(message, chat_history, model_name):
    qa_chain = create_rag_chain(model_name)
    result = qa_chain({"query": message})
    answer = result["result"] if isinstance(result, dict) else str(result)

    chat_history = chat_history or []
    # Добавляем в формате [user_msg, assistant_msg] для Chatbot
    chat_history.append([message, answer])

    return chat_history, chat_history

def respond(message, chat_history, model_name):
    chat_history, _ = chat_fn(message, chat_history, model_name)
    return "", chat_history

# ==========================
# Интерфейс Gradio
# ==========================
def build_interface():
    with gr.Blocks(title="RAG Chat") as demo:
        gr.Markdown("## 💬 RAG Chat for Codebase")

        with gr.Row():
            model_selector = gr.Dropdown(
                ["llama3.1:8b", "deepseek-coder:6.7b"],
                value=DEFAULT_LLM,
                label="Select Model"
            )
            clear_btn = gr.Button("🧹 Очистить историю")

        chatbot = gr.Chatbot(height=500)

        msg = gr.Textbox(
            placeholder="Введите вопрос о коде...",
            label="Ваш запрос"
        )
        send_btn = gr.Button("Отправить")

        state = gr.State([])  # хранит историю

        send_btn.click(
            respond,
            inputs=[msg, state, model_selector],
            outputs=[msg, chatbot]
        )

        msg.submit(
            respond,
            inputs=[msg, state, model_selector],
            outputs=[msg, chatbot]
        )

        clear_btn.click(lambda: [], None, chatbot)

    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="localhost", server_port=7860)
