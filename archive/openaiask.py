import os
import fitz  # PyMuPDF for text extraction
import faiss  # FAISS for similarity search
import numpy as np
import telebot
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pdfplumber  # Extract tables from PDFs
import pandas as pd  # Handle tables properly
from tiktoken import encoding_for_model

# Load environment variables
load_dotenv()

# Fetch API keys
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

if not api_key or not bot_token:
    raise ValueError("Environment variables for API keys not set.")

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


def count_tokens(text, model="gpt-4-turbo"):
    encoder = encoding_for_model(model)
    return len(encoder.encode(text))


def extract_text_and_tables(pdf_path):
    text = []
    tables = []
    
    # Extract text using PyMuPDF (fitz)
    try:
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text.append(page.get_text("text"))
    except Exception as e:
        print(f"Error extracting text with PyMuPDF: {e}")
    
    # Extract tables using pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_tables = page.extract_tables()
                for table in extracted_tables:
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        tables.append(df)
    except Exception as e:
        print(f"Error extracting tables with pdfplumber: {e}")

    full_text = "\n".join(text).strip()
    return full_text, tables


def load_knowledge_base():
    global documents, doc_texts, table_data
    documents, doc_texts, table_data = [], [], []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            text, tables = extract_text_and_tables(pdf_path)
            documents.append((pdf_file, text))
            doc_texts.append(text)
            table_data.extend(tables)
    print(f"Loaded {len(documents)} documents and {len(table_data)} tables.")


def load_custom_kb():
    global custom_kb
    if os.path.exists(kb_file):
        with open(kb_file, "r", encoding="utf-8") as f:
            custom_kb = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(custom_kb)} custom knowledge base entries.")


def save_custom_kb():
    with open(kb_file, "w", encoding="utf-8") as f:
        for entry in custom_kb:
            f.write(entry + "\n")
    print("Custom knowledge base saved.")


def create_faiss_index():
    global faiss_index, text_lengths
    all_texts = doc_texts + custom_kb
    text_lengths = [len(text.split()) for text in all_texts]
    if not all_texts:
        return None
    embeddings = embedding_model.encode(all_texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    print("FAISS index created with {} embeddings.".format(len(all_texts)))


load_knowledge_base()
load_custom_kb()
create_faiss_index()


def retrieve_context(query, top_k=5, max_tokens=6000):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    D, I = faiss_index.search(query_embedding, top_k)
    retrieved_texts = []
    total_tokens = 0
    for idx in I[0]:
        if idx < len(doc_texts):
            text = doc_texts[idx]
        else:
            text = custom_kb[idx - len(doc_texts)]
        text_tokens = count_tokens(text)
        if total_tokens + text_tokens > max_tokens:
            break
        retrieved_texts.append(text)
        total_tokens += text_tokens
    return "\n".join(retrieved_texts)


def ask_openai(prompt, model="gpt-4o", max_tokens=10000):
    total_tokens = count_tokens(prompt, model)
    if total_tokens > max_tokens:
        prompt_chunks = [prompt[i:i+max_tokens] for i in range(0, len(prompt), max_tokens)]
        responses = []
        for chunk in prompt_chunks:
            print(f"Sending chunk to OpenAI: {chunk}")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that answers questions based on provided knowledge."},
                    {"role": "user", "content": chunk}
                ]
            )
            responses.append(response.choices[0].message.content.strip())
        return " ".join(responses)
    else:
        print(f"Sending prompt to OpenAI: {prompt}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that answers questions based on provided knowledge."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()


def store_group_message(message):
    if len(group_messages) >= 100:
        group_messages.pop(0)
    group_messages.append(message.text)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello! Mention me with a question, and I'll answer using the organization's knowledge base.")


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_text = message.text
    if message.chat.type in ["group", "supergroup"]:
        store_group_message(message)
    is_mentioned = message.chat.type == "private"
    if message.chat.type in ["group", "supergroup"] and message.entities:
        for entity in message.entities:
            if entity.type == "mention":
                mentioned_text = user_text[entity.offset: entity.offset + entity.length]
                if mentioned_text.lower() == f"@{bot_username}":
                    is_mentioned = True
                    break
    if not is_mentioned:
        return
    if user_text.lower().startswith(f"@{bot_username} addkb") or (message.chat.type == "private" and user_text.lower().startswith("addkb")):
        new_kb_entry = user_text.replace(f"@{bot_username} addkb", "").strip() if message.chat.type != "private" else user_text.replace("addkb", "").strip()
        if new_kb_entry:
            custom_kb.append(new_kb_entry)
            save_custom_kb()
            create_faiss_index()
            bot.reply_to(message, "Knowledge added successfully!")
        return
    user_question = user_text.replace(f"@{bot_username}", "").strip()
    context = retrieve_context(user_question)
    if not context:
        bot.reply_to(message, "I couldn't find relevant information.")
        return
    prompt = f"Using the following knowledge, provide an answer:\n\nContext: {context}\n\nQuestion: {user_question}"
    print(f"Generated prompt: {prompt}")
    response = ask_openai(prompt)
    bot.reply_to(message, response if response else "I couldn't generate a response.")

bot.infinity_polling()