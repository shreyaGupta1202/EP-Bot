import os
import telebot
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import textwrap
import pdfplumber
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Knowledge base
pdf_folder = "data/pdf_docs"  # Folder to store PDFs
kb_file = "knowledge_base/kb.txt"  # File to store custom knowledge base
embeddings_cache_file = "embeddings_cache.json"  # File to cache embeddings
documents = []  # List to store PDF documents and their embeddings
custom_kb = []  # List to store custom knowledge base entries
custom_kb_embeddings = []  # List to store embeddings for custom_kb

# Ensure folders exist
os.makedirs(pdf_folder, exist_ok=True)
os.makedirs(os.path.dirname(kb_file), exist_ok=True)

### --- Load and Save Custom KB ---
def load_custom_kb():
    """Load custom knowledge base from file."""
    global custom_kb, custom_kb_embeddings
    if os.path.exists(kb_file):
        with open(kb_file, "r", encoding="utf-8") as f:
            custom_kb = [line.strip() for line in f.readlines() if line.strip()]
        # Generate embeddings for custom_kb
        custom_kb_embeddings = [
            genai.embed_content(
                model="models/embedding-001",
                content=entry,
                task_type="retrieval_document"
            )["embedding"]
            for entry in custom_kb
        ]

def save_custom_kb():
    """Save custom knowledge base to file."""
    with open(kb_file, "w", encoding="utf-8") as f:
        for entry in custom_kb:
            f.write(entry + "\n")

### --- Load and Save Embeddings Cache ---
def load_embeddings_cache():
    """Load embeddings cache from file."""
    if os.path.exists(embeddings_cache_file):
        with open(embeddings_cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_embeddings_cache(cache):
    """Save embeddings cache to file."""
    with open(embeddings_cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f)

### --- Split Text into Chunks ---
def split_text_into_chunks(text, max_chunk_size=9000):
    """
    Split text into chunks of a specified maximum size (in bytes).
    """
    chunks = []
    current_chunk = ""
    for sentence in text.split(". "):  # Split by sentences
        if len((current_chunk + sentence).encode("utf-8")) <= max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

### --- Extract Text from PDFs ---
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

### --- Generate Embeddings with Retries ---
def generate_embedding_with_retries(text, title, max_retries=3):
    """Generate embeddings with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            embedding = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document",
                title=title
            )["embedding"]
            return embedding
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    raise Exception("Failed to generate embedding after retries.")

### --- Load PDFs and Generate Embeddings ---
def load_pdfs_and_generate_embeddings():
    """Load PDFs from the folder and generate embeddings."""
    global documents
    documents = []
    cache = load_embeddings_cache()

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            if text:
                # Split text into chunks
                chunks = split_text_into_chunks(text)
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for chunk in chunks:
                        cache_key = f"{pdf_file}:{hash(chunk)}"
                        if cache_key in cache:
                            embedding = cache[cache_key]
                        else:
                            futures.append(executor.submit(generate_embedding_with_retries, chunk, pdf_file))
                    for future in as_completed(futures):
                        try:
                            embedding = future.result()
                            cache[cache_key] = embedding
                            documents.append({"title": pdf_file, "text": chunk, "embedding": embedding})
                        except Exception as e:
                            print(f"Failed to generate embedding: {e}")
    save_embeddings_cache(cache)

### --- Embedding Functions ---
def embed_query(query):
    """Generate embeddings for a query using Gemini API."""
    return genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )["embedding"]

### --- Find Best Passage ---
def find_best_passage(query):
    """
    Find the most relevant passage from the documents or custom knowledge base.
    """
    query_embedding = embed_query(query)
    best_score = -1
    best_text = ""

    # Search through PDF documents
    for doc in documents:
        doc_embedding = doc["embedding"]
        score = np.dot(query_embedding, doc_embedding)
        if score > best_score:
            best_score = score
            best_text = doc["text"]

    # Search through custom knowledge base
    for i, entry in enumerate(custom_kb):
        entry_embedding = custom_kb_embeddings[i]
        score = np.dot(query_embedding, entry_embedding)
        if score > best_score:
            best_score = score
            best_text = entry

    return best_text

### --- Make Prompt ---
def make_prompt(query, relevant_passage):
    """Generate a prompt for Gemini based on the query and relevant passage."""
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and conversational tone. \
    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """).format(query=query, relevant_passage=escaped)
    return prompt

### --- Load Knowledge Base and PDFs ---
load_custom_kb()
load_pdfs_and_generate_embeddings()

### --- Telegram Handlers ---
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello! Mention me with a question, and I'll answer using the organization's knowledge base.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_text = message.text

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
            # Generate embedding for the new entry
            new_embedding = genai.embed_content(
                model="models/embedding-001",
                content=new_kb_entry,
                task_type="retrieval_document"
            )["embedding"]
            custom_kb_embeddings.append(new_embedding)
            save_custom_kb()
            bot.reply_to(message, "Knowledge added successfully!")
        return

    # Remove bot mention in group chats
    user_question = user_text.replace(f"@{bot_username}", "").strip()

    # Retrieve context
    passage = find_best_passage(user_question)
    if not passage:
        bot.reply_to(message, "I couldn't find relevant information.")
        return

    # Ask Gemini AI
    prompt = make_prompt(user_question, passage)
    response = model.generate_content(prompt)
    bot.reply_to(message, response.text if response.text else "I couldn't generate a response.")

# Start polling
bot.infinity_polling()