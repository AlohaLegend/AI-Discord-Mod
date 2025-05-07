# 🤖 AI-Discord-Mod: Stanley 🤖

**Stanley** is a Discord bot that utilizes **OpenAI's Moderation API** for text moderation and **Hugging Face's Transformer model** for image moderation. Stanley helps maintain a safe and respectful environment in your Discord server.

This bot is **completely free** to use. OpenAI's Moderation API and Hugging Face's models are both free. Your OpenAI API key is only used to authenticate requests—you won't be charged for moderation usage.

<img width="919" alt="Screenshot" src="https://github.com/gravelBridge/AI-Discord-Mod/assets/107640947/95324ac4-5b76-4124-9679-e8f8dabb299d">

---

## 🌟 Features

- Text moderation using OpenAI's Moderation API
- Image moderation using Hugging Face’s transformer model
- Warns and mutes users for inappropriate messages
- Configurable thresholds, warning limits, and mute durations
- Admin-only commands to manage moderation behavior

---

## 🚀 Getting Started

Follow these steps to set up your own instance of **Stanley**:

### 📋 Prerequisites

Make sure you have the following installed:

- Python 3.6 or later  
- pip (Python package installer)

Install required packages:
```bash
pip install -r requirements.txt
pip install transformers datasets
pip install torch  # or pip install tensorflow
