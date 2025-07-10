import sqlite3
import os
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.error import TimedOut, RetryAfter
from dotenv import load_dotenv
import asyncio
import logging
import google.generativeai as genai
from textblob import TextBlob
from flask import Flask, request

# Setup Flask app for webhook
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RENDER_EXTERNAL_HOSTNAME = os.getenv("RENDER_EXTERNAL_HOSTNAME")

# Log environment variables for debugging
logging.info(f"TELEGRAM_TOKEN: {'Set' if TELEGRAM_TOKEN else 'Not set'}")
logging.info(f"GOOGLE_API_KEY: {'Set' if GOOGLE_API_KEY else 'Not set'}")
logging.info(f"RENDER_EXTERNAL_HOSTNAME: {RENDER_EXTERNAL_HOSTNAME or 'Not set'}")
logging.info(f"PORT: {os.getenv('PORT', '10000')}")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# System prompt
SYSTEM_PROMPT = """
You are CalmBot, an AI companion designed to help users heal from emotional distress and unresolved trauma, often rooted in childhood wounds. Your goal is to empower users to recognize, monitor, and heal deep-rooted emotional wounds, guiding them toward inner peace amidst external chaos. For every response:
- Acknowledge the user's emotion empathetically (e.g., happiness, sadness, anger, anxiety).
- Recognize potential trauma (e.g., childhood hurts, neglect) without assuming specifics.
- Use motivational, uplifting language to inspire resilience and hope (e.g., 'You're stronger than the scars you carry').
- Suggest actions like journaling, breathing exercises, or reflecting to find calm.
- Keep responses concise (100-150 words), empathetic, and warm, avoiding clinical tones.
- If the input is vague, ask a gentle follow-up question to clarify their needs.
"""

# Load model_log.json
try:
    with open("model_log.json", "r", encoding="utf-8") as f:
        DATASET = json.load(f)
    RESPONSE_MAP = {item["input"].lower(): item["output"] for item in DATASET}
except Exception as e:
    logging.error(f"Error loading model_log.json: {e}")
    RESPONSE_MAP = {}

# Initialize unknown_inputs.json
UNKNOWN_INPUTS_FILE = "unknown_inputs.json"
if not os.path.exists(UNKNOWN_INPUTS_FILE):
    with open(UNKNOWN_INPUTS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# SQLite database setup
def init_db():
    conn = sqlite3.connect("mood_tracker.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS responses
                 (user_id INTEGER, timestamp TEXT, mood TEXT, message TEXT)''')
    conn.commit()
    conn.close()

# Log unknown input
def log_unknown_input(user_id, user_message, is_followup=False):
    try:
        with open(UNKNOWN_INPUTS_FILE, "r", encoding="utf-8") as f:
            unknown_inputs = json.load(f)
    except:
        unknown_inputs = []
    unknown_inputs.append({
        "user_id": user_id,
        "timestamp": str(asyncio.get_event_loop().time()),
        "input": user_message,
        "is_followup": is_followup
    })
    with open(UNKNOWN_INPUTS_FILE, "w", encoding="utf-8") as f:
        json.dump(unknown_inputs, f, indent=2, ensure_ascii=False)

# Button responses
BUTTON_RESPONSES = {
    "happiness": "I feel the warmth of your happiness, like sunlight breaking through clouds! You're stronger than any scars you carry. To keep this joy flowing, try journaling what sparked this feeling today. Want to share more? Use /chat to dive deeper!",
    "sadness": "I hear the weight of your sadness, and it’s okay to feel this way sometimes. You’re not alone, and your heart is resilient. Try a deep breathing exercise: inhale for 4, hold for 4, exhale for 4. Want to talk more? Use /chat to explore what’s on your mind.",
    "anger": "Your anger is like a storm, powerful but temporary. You’re stronger than this moment. Try writing down what’s fueling it to let it go. Want to work through this together? Use /chat to share more and find calm.",
    "anxiety": "Anxiety can feel like a tight knot, but you’re stronger than the worries you carry. Try a quick grounding exercise: name 5 things you see around you. I’m here for you. Want to talk it out? Use /chat to share what’s weighing on you."
}

# Generate Gemini response
def generate_gemini_response(user_message, prev_response=None, conversation_history=""):
    try:
        full_prompt = f"{SYSTEM_PROMPT}\n\nConversation history: {conversation_history}\n\nPrevious bot response: {prev_response or 'None'}\n\nUser input: {user_message}\n\nRespond appropriately."
        response = gemini_model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        analysis = TextBlob(user_message)
        sentiment = 'positive' if analysis.sentiment.polarity > 0 else 'negative' if analysis.sentiment.polarity < 0 else 'neutral'
        return f"I hear you. Your message feels {sentiment}. Want to explore this further? Try sharing more or use /chat for support."

# Map input to response
def get_response(user_id, user_message, prev_response=None, context=None):
    user_message = user_message.lower().strip()
    if user_message in RESPONSE_MAP:
        if context:
            context.user_data["awaiting_followup"] = False
        return RESPONSE_MAP[user_message]
    if user_message == "yes" and prev_response:
        if context:
            context.user_data["awaiting_followup"] = False
        return generate_gemini_response(
            user_message,
            prev_response,
            context.user_data.get("conversation_history", "")
        )
    if context and context.user_data.get("awaiting_followup", False):
        log_unknown_input(user_id, user_message, is_followup=True)
        context.user_data["awaiting_followup"] = False
        return generate_gemini_response(
            user_message,
            prev_response,
            context.user_data.get("conversation_history", "")
        )
    log_unknown_input(user_id, user_message, is_followup=False)
    if context:
        context.user_data["awaiting_followup"] = True
    return generate_gemini_response(
        user_message,
        prev_response,
        context.user_data.get("conversation_history", "")
    )

# Log mood to database
def log_mood(user_id, mood, message):
    conn = sqlite3.connect("mood_tracker.db")
    c = conn.cursor()
    c.execute("INSERT INTO responses (user_id, timestamp, mood, message) VALUES (?, datetime('now'), ?, ?)",
              (user_id, mood, message))
    conn.commit()
    conn.close()

# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("😊 Happy", callback_data="happiness"),
         InlineKeyboardButton("😢 Sad", callback_data="sadness")],
        [InlineKeyboardButton("😡 Angry", callback_data="anger"),
         InlineKeyboardButton("😟 Anxious", callback_data="anxiety")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.user_data["awaiting_followup"] = False
    context.user_data["chat_mode"] = False
    context.user_data["conversation_history"] = ""
    context.user_data["prev_response"] = None
    await update.message.reply_text(
        "Hi! I’m CalmBot, your emotional support companion. Share how you feel, pick an emotion below, or use /chat to talk freely!",
        reply_markup=reply_markup
    )

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["chat_mode"] = True
    context.user_data["awaiting_followup"] = False
    await update.message.reply_text("Let’s chat! I’m here to listen and support you. What’s on your mind?")

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    mood = query.data
    user_id = query.from_user.id
    log_mood(user_id, mood, "Button selection")
    response = BUTTON_RESPONSES.get(mood, "I'm here to listen. Try /chat to talk freely.")
    context.user_data["prev_response"] = response
    context.user_data["awaiting_followup"] = False
    context.user_data["conversation_history"] = f"User selected mood: {mood} | Bot: {response} "
    keyboard = [
        [InlineKeyboardButton("Chat with CalmBot", callback_data="chat_after_mood"),
         InlineKeyboardButton("Change Response", callback_data="change_response")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.reply_text(response, reply_markup=reply_markup)

async def post_mood_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action = query.data
    if action == "chat_after_mood":
        context.user_data["chat_mode"] = True
        context.user_data["awaiting_followup"] = False
        await query.message.reply_text("Great, let’s dive deeper! What’s on your mind about how you’re feeling?")
    elif action == "change_response":
        keyboard = [
            [InlineKeyboardButton("😊 Happy", callback_data="happiness"),
             InlineKeyboardButton("😢 Sad", callback_data="sadness")],
            [InlineKeyboardButton("😡 Angry", callback_data="anger"),
             InlineKeyboardButton("😟 Anxious", callback_data="anxiety")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.user_data["awaiting_followup"] = False
        await query.message.reply_text("No worries! How are you feeling now?", reply_markup=reply_markup)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    user_id = update.message.from_user.id
    retries = 3
    for attempt in range(retries):
        try:
            if context.user_data.get("chat_mode", False):
                conversation_history = context.user_data.get("conversation_history", "")
                prev_response = context.user_data.get("prev_response", None)
                response = get_response(user_id, user_message, prev_response, context)
                log_mood(user_id, user_message.lower(), user_message)
                context.user_data["conversation_history"] = (conversation_history + f"User: {user_message} | Bot: {response} ")[-300:]
                context.user_data["prev_response"] = response
            else:
                response = get_response(user_id, user_message, context=context)
                log_mood(user_id, user_message.lower(), user_message)
                context.user_data["prev_response"] = response
                context.user_data["conversation_history"] = f"User: {user_message} | Bot: {response} "
            await update.message.reply_text(response)
            break
        except TimedOut:
            logging.warning(f"Timeout on attempt {attempt + 1}/{retries}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            logging.error("Max retries reached for timeout")
            await update.message.reply_text("Sorry, I’m having trouble connecting. Please try again later.")
            break
        except RetryAfter as e:
            logging.warning(f"Rate limit hit, retrying after {e.retry_after} seconds")
            await asyncio.sleep(e.retry_after)
            continue
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            await update.message.reply_text("An error occurred. Please try again.")
            break

# Root endpoint for debugging
@app.route('/')
def index():
    return "CalmBot is running! Use Telegram to interact with the bot.", 200

# Webhook test endpoint
@app.route('/test_webhook', methods=['GET'])
def test_webhook():
    webhook_url = f"https://{RENDER_EXTERNAL_HOSTNAME}/webhook"
    return f"Webhook URL: {webhook_url}. Check logs and Telegram getWebhookInfo.", 200

# Flask webhook endpoint
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        logging.info("Received webhook request")
        update = Update.de_json(request.get_json(force=True), app_telegram.bot)
        if update:
            logging.info(f"Processing update: {update}")
            asyncio.run(app_telegram.process_update(update))
            logging.info("Update processed successfully")
        else:
            logging.warning("No valid update received")
        return '', 200
    except Exception as e:
        logging.error(f"Webhook error: {str(e)}")
        return '', 500

async def set_webhook():
    webhook_url = f"https://{RENDER_EXTERNAL_HOSTNAME}/webhook"
    try:
        await app_telegram.bot.set_webhook(webhook_url)
        logging.info(f"Webhook set to {webhook_url}")
    except Exception as e:
        logging.error(f"Failed to set webhook: {str(e)}")

# Initialize Telegram app
app_telegram = Application.builder().token(TELEGRAM_TOKEN).updater(None).build()
app_telegram.add_handler(CommandHandler("start", start))
app_telegram.add_handler(CommandHandler("chat", chat))
app_telegram.add_handler(CallbackQueryHandler(button, pattern='^(happiness|sadness|anger|anxiety)$'))
app_telegram.add_handler(CallbackQueryHandler(post_mood_button, pattern='^(chat_after_mood|change_response)$'))
app_telegram.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

if __name__ == "__main__":
    init_db()
    # Set webhook on startup
    asyncio.run(set_webhook())
    # Run Flask app
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
