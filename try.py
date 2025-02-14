import os
import fitz  # PyMuPDF for PDF text extraction
import faiss  # FAISS for similarity search
import numpy as np
import telebot
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Fetch API keys
api_key = os.getenv("GOOGLE_API_KEY")
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

# Validate API keys
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
if not bot_token:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set.")

# Configure Gemini AI
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-pro")

# Initialize Telegram Bot
bot = telebot.TeleBot(bot_token)
bot_username = bot.get_me().username.lower()

# Load Sentence Transformer for embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# --- Step 1: Load PDFs and Create an Index ---
pdf_folder = "pdf_docs"
documents = []
doc_texts = []

# Extract text from all PDFs
def load_pdfs():
    global documents, doc_texts
    documents = []
    doc_texts = []

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            with fitz.open(pdf_path) as doc:
                text = "\n".join([page.get_text("text") for page in doc])
                documents.append((pdf_file, text))
                doc_texts.append(text)

    print(f"Loaded {len(documents)} PDFs into knowledge base.")

# Generate embeddings for stored text
def create_faiss_index():
    global faiss_index
    if not doc_texts:
        return None

    embeddings = embedding_model.encode(doc_texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

load_pdfs()
create_faiss_index()

# --- Step 2: Retrieve Relevant Content ---
def retrieve_context(query, top_k=2):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    
    D, I = faiss_index.search(query_embedding, top_k)
    retrieved_texts = [doc_texts[idx] for idx in I[0] if idx < len(doc_texts)]
    
    return "\n".join(retrieved_texts) if retrieved_texts else None

# --- Telegram Handlers ---
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello! Mention me in a group chat with a question, and I'll answer based on the organization's documents.")

@bot.message_handler(func=lambda message: True)
def ask_gemini(message):
    user_question = message.text

    # --- ✅ FIX: Ensure bot only responds when tagged ---
    if message.chat.type in ["group", "supergroup"]:
        is_mentioned = False

        if message.entities:
            for entity in message.entities:
                if entity.type == "mention":
                    mentioned_text = user_question[entity.offset: entity.offset + entity.length]
                    if mentioned_text.lower() == f"@{bot_username}":
                        is_mentioned = True
                        break
        
        if not is_mentioned:
            return  # Ignore messages where the bot isn't mentioned

    # Remove mention from the query
    user_question = user_question.replace(f"@{bot_username}", "").strip()

    # Retrieve relevant context from PDFs
    context = retrieve_context(user_question)
    
    if not context:
        bot.reply_to(message, "I couldn't find relevant information in the documents.")
        return

    # Ask Gemini with retrieved context
    prompt = f"Based on the organization's knowledge base, answer the following:\n\n{context}\n\nQuestion: {user_question}"
    
    try:
        response = model.generate_content(prompt)
        gemini_answer = response.text if response.text else "I couldn't generate a response."
        bot.reply_to(message, gemini_answer)
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        bot.reply_to(message, "Sorry, I'm having trouble answering that right now.")

# Start polling
bot.infinity_polling()
