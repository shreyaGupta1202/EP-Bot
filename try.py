import os
import fitz  # PyMuPDF for text extraction
import faiss  # FAISS for similarity search
import numpy as np
import telebot
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pdfplumber  # Extract tables from PDFs
import pandas as pd  # Handle tables properly

# Load environment variables
load_dotenv()

# Fetch API keys
api_key = os.getenv("GOOGLE_API_KEY")
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

# Validate API keys
if not api_key or not bot_token:
    raise ValueError("Environment variables for API keys not set.")

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
table_data = []  # Stores tables separately for better querying
custom_kb = []  # List for additional knowledge base entries
group_messages = []  # Store last 100 group messages


### --- Extract Text & Tables from PDFs ---
def extract_text_and_tables(pdf_path):
    text = []
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")

            # Extract tables and convert them to structured DataFrames
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                if table:  # Ensure table is not empty
                    df = pd.DataFrame(table[1:], columns=table[0])  # Convert table to DataFrame
                    tables.append(df)

    text_content = "\n".join(text)

    return text_content, tables


### --- Load PDFs and Extract Knowledge ---
def load_knowledge_base():
    global documents, doc_texts, table_data
    documents = []
    doc_texts = []
    table_data = []

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            text, tables = extract_text_and_tables(pdf_path)
            documents.append((pdf_file, text))
            doc_texts.append(text)

            # Store structured table data separately
            for df in tables:
                table_data.append(df)


### --- Load and Save Custom KB ---
def load_custom_kb():
    global custom_kb
    if os.path.exists(kb_file):
        with open(kb_file, "r", encoding="utf-8") as f:
            custom_kb = [line.strip() for line in f.readlines() if line.strip()]


def save_custom_kb():
    with open(kb_file, "w", encoding="utf-8") as f:
        for entry in custom_kb:
            f.write(entry + "\n")


### --- FAISS Index for Similarity Search ---
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


### --- Context Retrieval (Fix for Attendees) ---
def retrieve_context(query, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    D, I = faiss_index.search(query_embedding, top_k)
    retrieved_texts = [doc_texts[idx] if idx < len(doc_texts) else custom_kb[idx - len(doc_texts)] for idx in I[0] if idx < len(doc_texts) + len(custom_kb)]

    combined_text = "\n".join(retrieved_texts) if retrieved_texts else ""

    # Special handling for attendee-related questions
    if "attendees" in query.lower() or "present" in query.lower():
        attendees = []
        for df in table_data:  # Iterate over stored tables
            if "Present" in df.columns:  # Check if the "Present" column exists
                present_rows = df[df["Present"].str.strip().str.upper().isin(["Y", "YES"])]
                if not present_rows.empty:
                    attendees.append(present_rows.to_string(index=False))

        return "\n".join(attendees) if attendees else "No attendees marked as present."

    return combined_text


### --- Store Last 100 Messages ---
def store_group_message(message):
    if len(group_messages) >= 100:
        group_messages.pop(0)
    group_messages.append(message.text)


### --- Telegram Handlers ---
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello! Mention me with a question, and I'll answer using the organization's knowledge base.")


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_text = message.text

    # Store message for retrieval in groups
    if message.chat.type in ["group", "supergroup"]:
        store_group_message(message)

    # Check if the bot is mentioned in group chats
    is_mentioned = message.chat.type == "private"  # Always respond in DMs

    if message.chat.type in ["group", "supergroup"] and message.entities:
        for entity in message.entities:
            if entity.type == "mention":
                mentioned_text = user_text[entity.offset: entity.offset + entity.length]
                if mentioned_text.lower() == f"@{bot_username}":
                    is_mentioned = True
                    break

    # Ignore messages in group chats unless bot is mentioned
    if not is_mentioned:
        return

    # Handle Knowledge Base Addition
    if user_text.lower().startswith(f"@{bot_username} addkb") or (message.chat.type == "private" and user_text.lower().startswith("addkb")):
        new_kb_entry = user_text.replace(f"@{bot_username} addkb", "").strip() if message.chat.type != "private" else user_text.replace("addkb", "").strip()
        if new_kb_entry:
            custom_kb.append(new_kb_entry)
            save_custom_kb()
            create_faiss_index()
            bot.reply_to(message, "Knowledge added successfully!")
        return

    # Remove bot mention in group chats
    user_question = user_text.replace(f"@{bot_username}", "").strip()

    # Retrieve context
    context = retrieve_context(user_question)

    if not context:
        bot.reply_to(message, "I couldn't find relevant information.")
        return

    # Ask Gemini AI
    prompt = f"Using the following knowledge, provide an answer:\n\nContext: {context}\n\nQuestion: {user_question}"
    response = model.generate_content(prompt)
    bot.reply_to(message, response.text if response.text else "I couldn't generate a response.")



# Start polling
bot.infinity_polling()
