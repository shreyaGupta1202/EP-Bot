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

# Knowledge base
pdf_folder = "pdf_docs"
kb_file = "knowledge_base.txt"
documents = []
doc_texts = []
custom_kb = []  # List for additional knowledge base entries

# Load custom knowledge base from file
def load_custom_kb():
    global custom_kb
    if os.path.exists(kb_file):
        with open(kb_file, "r", encoding="utf-8") as f:
            custom_kb = [line.strip() for line in f.readlines() if line.strip()]

def save_custom_kb():
    with open(kb_file, "w", encoding="utf-8") as f:
        for entry in custom_kb:
            f.write(entry + "\n")

# Load PDFs and .txt files
def load_knowledge_base():
    global documents, doc_texts
    documents = []
    doc_texts = []
    
    # Load PDFs
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            with fitz.open(pdf_path) as doc:
                text = "\n".join([page.get_text("text") for page in doc])
                documents.append((pdf_file, text))
                doc_texts.append(text)
    
    # Load .txt file
    if os.path.exists(kb_file):
        with open(kb_file, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append((kb_file, text))
            doc_texts.append(text)

# Generate embeddings for stored text
def create_faiss_index():
    global faiss_index
    all_texts = doc_texts + custom_kb
    if not all_texts:
        return None
    embeddings = embedding_model.encode(all_texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

load_knowledge_base()
load_custom_kb()
create_faiss_index()

# Retrieve relevant context
def retrieve_context(query, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    D, I = faiss_index.search(query_embedding, top_k)
    retrieved_texts = [doc_texts[idx] if idx < len(doc_texts) else custom_kb[idx - len(doc_texts)] for idx in I[0] if idx < len(doc_texts) + len(custom_kb)]
    return "\n".join(retrieved_texts) if retrieved_texts else None

# Store last 100 group messages
group_messages = []

def store_group_message(message):
    if len(group_messages) >= 100:
        group_messages.pop(0)
    group_messages.append(message.text)

# Telegram Handlers
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello! Mention me with a question, and I'll answer using the organization's knowledge base.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_text = message.text

    # Store message for retrieval
    if message.chat.type in ["group", "supergroup"]:
        store_group_message(message)
        
    # Ensure bot only responds when tagged
    is_mentioned = False
    if message.chat.type in ["group", "supergroup"] and message.entities:
        for entity in message.entities:
            if entity.type == "mention":
                mentioned_text = user_text[entity.offset: entity.offset + entity.length]
                if mentioned_text.lower() == f"@{bot_username}":
                    is_mentioned = True
                    break
    
    if not is_mentioned:
        return

    # Check if it's an addkb command
    if user_text.lower().startswith(f"@{bot_username} addkb"):
        new_kb_entry = user_text.replace(f"@{bot_username} addkb", "").strip()
        if new_kb_entry:
            custom_kb.append(new_kb_entry)
            save_custom_kb()
            create_faiss_index()
            bot.reply_to(message, "Knowledge added successfully!")
        return

    # Remove bot mention from user query
    user_question = user_text.replace(f"@{bot_username}", "").strip()
    
    # Retrieve from PDFs, .txt file, and additional KB
    context = retrieve_context(user_question)
    if not context:
        # Check last 100 messages with similarity search
        msg_embeddings = embedding_model.encode(group_messages, convert_to_numpy=True)
        user_embedding = embedding_model.encode([user_question], convert_to_numpy=True)
        D, I = faiss.IndexFlatL2(msg_embeddings.shape[1]).search(user_embedding, 3)
        relevant_msgs = [group_messages[idx] for idx in I[0] if idx < len(group_messages)]
        if relevant_msgs:
            context = "\n".join(relevant_msgs)
    
    if not context:
        bot.reply_to(message, "I couldn't find relevant information in the documents or recent messages.")
        return
    
    # Ask Gemini with retrieved context
    prompt = f"Using the following knowledge sources, provide an informative and well-analyzed response:\n\nContext: {context}\n\nQuestion: {user_question}"
    try:
        response = model.generate_content(prompt)
        bot.reply_to(message, response.text if response.text else "I couldn't generate a response.")
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        bot.reply_to(message, "Sorry, I'm having trouble answering that right now.")

# Start polling
bot.infinity_polling()