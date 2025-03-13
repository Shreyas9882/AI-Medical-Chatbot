from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM  # ✅ Correct Import
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Load Environment Variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ✅ Download Hugging Face Embeddings
embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

# ✅ Load Existing Pinecone Index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Initialize Ollama Model (Streaming Enabled)
llm = OllamaLLM(model="mistral", streaming=True)  # ✅ Live Streaming for Fast Responses

# ✅ Define Chat Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# ✅ Create Retrieval-Augmented Generation (RAG) Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ✅ Store Chat History (Per Session)
chat_history = []

# ✅ Flask Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a message."  # ✅ Returns plain text

    print(f"User Input: {msg}")

    # Add user message to chat history
    chat_history.append({"role": "user", "content": msg})

    try:
        # Get response from RAG model
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "I'm sorry, I couldn't process your request.")
    except Exception as e:
        print(f"Error: {e}")
        answer = "An error occurred while processing your request."

    # Add AI response to chat history
    chat_history.append({"role": "bot", "content": answer})

    print(f"Response: {answer}")
    
    return answer  # ✅ Returns only plain text (No JSON)

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
