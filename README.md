# BrokeNoMore

A personal expense tracker I built to stop wondering where my money goes every month. It runs locally on your machine, stores everything in a local database, and uses machine learning to get smarter about your spending habits over time.

---

## What it does

- Tracks expenses across multiple payment methods (Cash, UPI, Bank, Credit Card)
- Automatically understands the category of an expense from what you type — "filled up the activa", "fuel bunk", "petrol for bike" all correctly map to Transport
- Detects when you've spent unusually more than usual in any category
- Predicts how much you'll spend by end of month based on your current pace
- Groups your spending into patterns using clustering
- Gives you a daily budget based on what's left
- Tracks recurring subscriptions separately
- Tracks income and shows your net balance for the month
- Lets you set saving challenges and earn points for completing them
- Has an AI advisor you can ask anything — "I have ₹5000 left for 12 days, how do I manage?"
- Paisa the owl — a mascot that walks around the screen, watches your spending, and says something about it every time you log an expense. Powered by Llama so the reactions are always fresh and specific to what you actually did
- You can upload a photo of a bill and it'll read the amount and details automatically

---

## Tech used

- Python + Flask for the backend
- scikit-learn for anomaly detection, forecasting, and clustering
- Llama 3.1 (8B) via Ollama for smart categorization, AI advisor, and mascot reactions — fully offline, runs on GPU
- SQLite for local data storage
- Vanilla HTML/CSS/JS for the frontend

---

## Setup

Make sure you have Python 3.9 or above installed.

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

On Windows you can just double-click `START.bat`.

---

## AI + Ollama setup (one time)

The categorizer, AI advisor, and mascot all run on Llama 3.1 locally. No API key, no internet needed after setup.

**Step 1 — Install Ollama:**
Download from ollama.com and run the installer.

**Step 2 — Download the model:**
```cmd
ollama pull llama3.1
```
Downloads ~4.9GB once. Available offline forever after that.

**That's it.** The app auto-starts Ollama every time you run it. You'll see a confirmation in the terminal.

Tested on RTX 4050 (6GB VRAM) — categorization takes ~2 seconds, mascot reactions ~3 seconds, AI advisor ~5-10 seconds.

---

## Notes

- All data stays on your machine, nothing is sent anywhere
- The categorizer uses Llama to understand context, not just keywords — so varied descriptions of the same thing get categorized correctly
- Tested on Windows 11

---

## Why I built this

I kept going over budget without realizing it until the end of the month. Most expense apps are either too simple or require a subscription themselves, which felt ironic. Wanted something that actually learns from how I spend rather than just logging numbers.
