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
- Lets you set saving challenges and earn points for completing them
- Has an AI advisor you can ask things like "I have ₹5000 left for 12 days, how do I manage?"
- You can upload a photo of a bill and it'll read the amount and details automatically

---

## Tech used

- Python + Flask for the backend
- scikit-learn for the ML models (Naive Bayes for categorization, Linear Regression for forecasting, K-Means for clustering, z-score for anomaly detection)
- SQLite for local data storage
- Claude API for the AI advisor and receipt scanning (optional)
- Vanilla HTML/CSS/JS for the frontend

---

## Setup

Make sure you have Python 3.9 or above installed.

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

On Windows you can just double-click `START.bat` and it handles everything.

---

## AI features (optional)

The AI advisor and receipt scanner use the Anthropic API. If you want those, set your API key before running:

```bash
set ANTHROPIC_API_KEY=your_key_here
```

Everything else works fine without it.

---

## Notes

- All your data stays on your machine, nothing is sent anywhere (except the AI calls if you use that feature)
- The categorizer learns from your own expenses over time, so it gets more accurate the more you use it
- Tested on Windows 11

---

## Why I built this

I kept going over budget without realizing it until the end of the month. Most expense apps are either too simple or require a subscription themselves, which felt ironic. Wanted something that actually learns from how I spend rather than just logging numbers.
