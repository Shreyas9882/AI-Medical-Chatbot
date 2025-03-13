# AI Powered Medical Chatbot

## Introduction
The **Medical Chatbot** is an AI-powered conversational agent designed to provide medical-related information and assist users with health inquiries. This chatbot leverages Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) to deliver accurate and context-aware responses. Built using **Flask, Ollama, LangChain, and Pinecone**, it ensures fast and efficient retrieval of medical knowledge.

## Project Methodology 

### 1. Technology Stack
- **Frontend**: HTML, CSS, Bootstrap
- **Backend**: Flask (Python), LangChain, Ollama (LLM), Pinecone (Vector Database)
- **Embeddings**: Hugging Face embeddings for document search
- **LLM**: Mistral model (via Ollama) for AI responses
- **Database**: Pinecone for efficient medical data retrieval

### 2. Workflow
1. **User Interaction**: The user enters a medical query in the chatbot UI.
2. **Query Processing**: The query is sent to the Flask backend.
3. **Retrieval-Augmented Generation (RAG)**:
   - The system fetches relevant medical information from Pinecone.
   - The retrieved data is processed using the Ollama LLM.
   - A chatbot response is generated using a structured chat prompt.
4. **Response Delivery**: The chatbot displays the AI-generated answer in the UI.
5. **Chat History Maintenance**: The chatbot maintains a session-based history for contextual responses.

## Project Implementatiomm

### 1. Prerequisites
Ensure the following dependencies are installed:
- Python 3.8+
- Flask
- LangChain
- Pinecone
- Ollama
- Hugging Face transformers
- dotenv (for environment variables)

### 2. Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/medical-chatbot.git
   cd medical-chatbot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file in the project root.
   - Add your Pinecone API key:
     ```
     PINECONE_API_KEY=your_api_key_here
     ```
4. Run the Flask server:
   ```bash
   python app.py
   ```
5. Open the chatbot in your browser:
   ```
   http://localhost:5000
   ```

## Project Excecution
<img width="1440" alt="Image" src="https://github.com/user-attachments/assets/140e3abc-25ea-4403-b7a5-35d84f11e389" />
<img width="1440" alt="Image" src="https://github.com/user-attachments/assets/63ab691e-25b8-4c35-9994-79596d4ea909" />
<img width="1440" alt="Image" src="https://github.com/user-attachments/assets/569eb2d5-bd0c-4b10-b11b-e12c3cf25032" />

### 3. Usage
- Type a medical query in the chatbot interface.
- The chatbot will analyze the query and provide a relevant response.
- Continue the conversation to get additional insights.

## Conclusion
This Medical Chatbot provides an efficient and intelligent way to assist users with health-related queries. By integrating advanced AI and retrieval techniques, it ensures accurate and real-time responses. Future enhancements could include voice support, multilingual capabilities, and integration with healthcare APIs.

