# 🆘 Emergency Response Chatbot 🚨

A Telegram-based AI-powered emergency assistance chatbot that helps users quickly report incidents such as medical emergencies, fires, crimes, or natural disasters by collecting relevant data like emergency type, user location, additional address, and incident images. The bot notifies admins with a complete report and media via Telegram.

---

## 🔍 Features

- 🌐 **Conversational AI** using LangChain & Google Gemini (via `langchain_google_genai`)
- 📍 **Geolocation detection** and reverse geocoding using GeoPy
- 🧠 **Emergency type detection** using custom NLP logic (with NLTK)
- 🖼️ **Image upload support** for incident evidence
- 📬 **Admin notification system** with maps and media group
- ✅ **Interactive options** (Ambulance, Police, Fire, Disaster) with inline keyboards
- 🔐 Environment variable support using `python-dotenv`

---

## 📦 Tech Stack

| Layer         | Tools/Packages                                 |
|---------------|------------------------------------------------|
| Language      | Python                                         |
| Bot Framework | `python-telegram-bot`                          |
| AI/NLP        | LangChain, Google Gemini Pro, NLTK             |
| Location      | GeoPy, Nominatim                               |
| UI            | Telegram InlineKeyboard                        |
| Data Handling | Regex, POS tagging, Named Entity Recognition   |
| Deployment    | `async`/`await`-based bot via polling          |

---

## 🚀 Getting Started

### 📁 Clone the Repository
```bash
git clone https://github.com/your-username/emergency-chatbot.git
cd emergency-chatbot

---

###Install Dependencies
pip install -r requirements.txt
