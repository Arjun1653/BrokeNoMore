# BrokeNoMore

A personal expense tracker I built to stop wondering where my money goes every month. It runs locally on your machine, stores everything in a local database, and uses machine learning to get smarter about your spending habits over time.

---

## What it does

- Tracks expenses across multiple payment methods (Cash, UPI, Bank, Credit Card)
- Automatically figures out the category of an expense based on what you type
- Detects when you've spent unusually more than usual in any category
- Predicts how much you'll spend by end of month based on your current pace
- Groups your spending into patterns using clustering
- Gives you a daily budget based on what's left
- Tracks recurring subscriptions separately
- Tracks income and shows your net balance for the month
- Lets you set saving challenges and earn points for completing them
- Has an AI advisor you can ask things like "I have ₹5000 left for 12 days, how do I manage?"
- You can upload a photo of a bill and it'll read the amount and details automatically

---

## Tech used

- Python + Flask for the backend
- scikit-learn for the ML models (Naive Bayes for categorization, Linear Regression for forecasting, K-Means for clustering, z-score for anomaly detection)
- SQLite for local data storage
- Llama 3.1 (8B) via Ollama for the AI advisor — runs fully offline on your GPU
- Vanilla HTML/CSS/JS for the frontend

---

## Setup

Make sure you have Python 3.9 or above installed.

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

---

## AI Advisor setup (one time)

The AI advisor runs locally using Llama 3.1 through Ollama. No API key, no internet needed after setup.

**Step 1 — Install Ollama:**
Download from ollama.com and run the installer.

**Step 2 — Download the model:**
```cmd
ollama pull llama3.1
```
This downloads the 4.9GB model once. After that it's available offline forever.

Ollama runs automatically in the background after installation, so no extra steps are needed each time you use the app. Just start the app and the AI Advisor will work.

Tested with an RTX 4050 (6GB VRAM) — responses come back in a few seconds.

---

## Notes

- All your data stays on your machine, nothing is sent anywhere
- The categorizer learns from your own expenses over time, so it gets more accurate the more you use it
- Tested on Windows 11

---

## Why I built this

I kept going over budget without realizing it until the end of the month. Most expense apps are either too simple or require a subscription themselves, which felt ironic. Wanted something that actually learns from how I spend rather than just logging numbers.
