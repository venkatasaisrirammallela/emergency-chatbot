import os
import logging
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from functools import lru_cache
import nltk
import re
from geopy.geocoders import Nominatim

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load environment variables
load_dotenv()

# Telegram Bot Token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Gemini API Token
GEMINI_API_TOKEN = os.getenv("GEMINI_API_TOKEN")

# Admin Chat ID
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

# Configure logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Template for prompting responses
prompt_template = """
As an emergency assistance chatbot, provide a concise, precise, and detailed response to the question, while maintaining a calm, reassuring, and engaging tone. If the provided context is relevant to the emergency situation, incorporate information from it into your response. However, if the question is not directly related to the context, draw upon your knowledge and expertise to provide helpful guidance or advice related to emergency preparedness and response.

Context:{context}

Question:{question}

Answer:
"""

@lru_cache(maxsize=None)
def setup_conversational_chain():
    # Initialize the chatbot model
    model = ChatGoogleGenerativeAI(model="gemini-pro", api_key=GEMINI_API_TOKEN, temperature=0.7)
    # Set up the prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Load the conversational chain
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Initialize the conversational chain
chain = setup_conversational_chain()

# Command handler for '/start' command
async def start(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    await update.message.reply_html(rf"Hello {user.mention_html()}, I am an Emergency Registration Chatbot. What's your emergency? Please select an option or describe your emergency.")
    options = ["Ambulance", "Police", "Fire Brigade", "Disaster Management"]
    await present_options(update, context, options)

# Handler for receiving user location
async def location(update: Update, context: CallbackContext) -> None:
    user_location = update.message.location
    if user_location:
        context.user_data['location'] = user_location
        geolocator = Nominatim(user_agent="Emergency Bot")
        location = geolocator.reverse(f"{user_location.latitude}, {user_location.longitude}", exactly_one=True)
        location_address = location.address
        await update.message.reply_text(f"Location: {location_address}")
        await update.message.reply_text("Do you want to add any additional address details?", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Yes", callback_data="add_address"), InlineKeyboardButton("No", callback_data="skip_address")]]))
    else:
        await update.message.reply_text("Sorry, I couldn't retrieve your location.")

# Handler for receiving user's address choice
async def handle_address_choice(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()
    if query.data == "add_address":
        await query.edit_message_text("Please enter any additional address details.")
        context.user_data['address_choice'] = "add"
    else:
        context.user_data['address_choice'] = "skip"
        await query.edit_message_text("Understood, let's proceed.")
        await query.message.reply_text("Please upload at least one image regarding the emergency.")

# Handler for receiving additional address details
async def handle_address(update: Update, context: CallbackContext) -> None:
    if context.user_data.get('address_choice') == "add":
        additional_address = update.message.text
        context.user_data['additional_address'] = additional_address
        await update.message.reply_text("Thank you for the additional address details. Please upload at least one image regarding the emergency.")
    else:
        await update.message.reply_text("Please upload at least one image regarding the emergency.")

# Handler for receiving images
async def handle_image(update: Update, context: CallbackContext) -> None:
    images = update.message.photo
    if images:
        media = [InputMediaPhoto(media=img.file_id, caption="Emergency Image") for img in images]
        await send_admin_message(update, context, media)
        await update.message.reply_text("Thank you for sharing. An emergency response team will be there shortly.")
    else:
        await update.message.reply_text("Please upload at least one image.")

# Handler for receiving text messages
async def handle_text(update: Update, context: CallbackContext) -> None:
    user_text = update.message.text
    if context.user_data.get('address_choice') == "add":
        additional_address = user_text
        context.user_data['additional_address'] = additional_address
        await update.message.reply_text("Thank you for the additional address details. Please upload at least one image regarding the emergency.")
        context.user_data['address_choice'] = None
    else:
        if user_text:
            # Analyze the text using NLP
            emergency_types = analyze_text(user_text)
            if emergency_types:
                context.user_data['selected_options'] = emergency_types
                options_text = ", ".join(emergency_types)
                await update.message.reply_text(f"Understood, you need {options_text} assistance. Please share your location.")
            else:
                await update.message.reply_text("I'm sorry, I couldn't understand your emergency. Please try again or select an option from the menu.")

# Function to present options to the user
async def present_options(update: Update, context: CallbackContext, options: list) -> None:
    keyboard = [[InlineKeyboardButton(option, callback_data=option)] for option in options]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Please select an option:", reply_markup=reply_markup)

# Handler for user's option selection
async def handle_option_selection(update: Update, context: CallbackContext) -> None:
    selected_option = update.callback_query.data
    current_options = context.user_data.get('selected_options', [])
    if selected_option not in current_options:
        current_options.append(selected_option)
        context.user_data['selected_options'] = current_options
    await update.callback_query.answer()
    options_text = ", ".join(current_options)
    await update.callback_query.edit_message_text(f"You selected: {options_text}")
    await update.callback_query.message.reply_text("Please share your location.")

# Function to send admin message and images
async def send_admin_message(update: Update, context: CallbackContext, media: list) -> None:
    selected_options = context.user_data.get('selected_options')
    location = context.user_data.get('location')
    additional_address = context.user_data.get('additional_address', '')

    geolocator = Nominatim(user_agent="Emergency Bot")
    location_address = geolocator.reverse(f"{location.latitude}, {location.longitude}", exactly_one=True).address
    location_link = f"https://www.google.com/maps/search/?api=1&query={location.latitude},{location.longitude}"

    admin_message = f"Emergency: {', '.join(selected_options)}\nLocation: {location_address}\nAdditional Address: {additional_address}\nGoogle Maps: {location_link}"
    await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=admin_message)

    # Sending images
    if media:
        await context.bot.send_media_group(chat_id=ADMIN_CHAT_ID, media=media)

    # Send a friendly message to the user
    await update.effective_chat.send_message("Thank you for reporting the emergency. Help is on the way! Stay safe, and don't hesitate to provide any additional information if needed.")

# Function to analyze text for emergency type
def analyze_text(text):
    # Preprocess the text
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    tagged_tokens = nltk.pos_tag(filtered_tokens)

    # Named Entity Recognition
    entities = nltk.ne_chunk(tagged_tokens)

    # Define emergency keyword patterns
    emergency_keywords = {
        'ambulance': ['accident', 'injury', 'medical', 'hurt', 'sick', 'health', 'pain', 'emergency'],
        'police': ['crime', 'robbery', 'theft', 'violence', 'assault', 'law', 'illegal', 'arrest'],
        'fire brigade': ['fire', 'blaze', 'flames', 'smoke', 'burning'],
        'disaster management': ['disaster', 'earthquake', 'flood', 'hurricane', 'tornado']
    }

    # Pattern matching
    emergency_patterns = [
        r'\b(accident|injury|medical|hurt|sick|health|pain|emergency)\b',
        r'\b(crime|robbery|theft|violence|assault|law|illegal|arrest)\b',
        r'\b(fire|blaze|flames|smoke|burning)\b',
        r'\b(disaster|earthquake|flood|hurricane|tornado)\b'
    ]

    emergency_types = []
    for entity in entities:
        if hasattr(entity, 'label') and entity.label() == 'PERSON':
            continue  # Skip person entities
        for emergency_type, keywords in emergency_keywords.items():
            for keyword in keywords:
                if keyword in entity:
                    emergency_types.append(emergency_type)
                    break

    for pattern in emergency_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            for emergency_type, keywords in emergency_keywords.items():
                for keyword in keywords:
                    if re.search(r'\b{}\b'.format(keyword), text, re.IGNORECASE):
                        emergency_types.append(emergency_type)
                        break

    if 'fire brigade' in emergency_types:
        emergency_types.append('ambulance')

    return list(set(emergency_types))

# Main function
def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.LOCATION, location))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(CallbackQueryHandler(handle_option_selection, pattern=r'^(Ambulance|Police|Fire Brigade|Disaster Management)$'))
    application.add_handler(CallbackQueryHandler(handle_address_choice, pattern=r'^(add_address|skip_address)$'))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_address))
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()