# 💰 Smart Expense Tracker — ML Powered

A fully local, ML-powered personal finance app built with Flask + scikit-learn + Claude AI.

---

## 🚀 Quick Start (Windows)

### Option 1: Double-click launcher
```
Double-click START.bat
```
Then open http://127.0.0.1:5000 in your browser.

### Option 2: Manual
```bash
pip install -r requirements.txt
python app.py
```

---

## 🤖 AI Features Setup

For AI Advisor and Receipt Scanning, set your Anthropic API key:

**Windows (this session):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-...your-key...
python app.py
```

**Windows (permanent):**
1. Search "Environment Variables" in Start menu
2. Add `ANTHROPIC_API_KEY` = your key
3. Restart terminal

Get your key at: https://console.anthropic.com

> Note: All ML features (categorization, anomaly detection, forecasting, clustering, habit score) work **without** the API key. Only AI Advisor chat and Receipt Scanning need it.

---

## 🧠 ML Models Used

| Feature | Algorithm |
|---------|-----------|
| Auto-categorization | Naive Bayes (MultinomialNB) with TF-IDF |
| Anomaly detection | Statistical z-score + Isolation Forest |
| Spending forecast | Linear Regression on time series |
| Spending clusters | K-Means clustering |
| Habit scoring | ML feature extraction + rule engine |
| AI Advice | Anthropic Claude (via API) |
| Receipt scanning | Claude Vision (via API) |
| NL queries | Rule-based NLP on SQLite data |

The categorizer **learns from your data** — the more you log, the smarter it gets.

---

## ✨ Features

- **Dashboard** — balance, budget progress, habit score, alerts
- **Add Expense** — manual entry with ML auto-categorization, receipt scanning
- **Insights** — category breakdown, daily chart, anomaly detection, K-Means clusters, 30-day forecast
- **AI Advisor** — Claude-powered financial advice with full context
- **Subscriptions** — monthly subscription tracker with billing days
- **Challenges** — gamified saving goals with points
- **Wallets** — Cash, Bank, UPI, Credit Card tracking
- **Export** — CSV download of all expenses
- **Natural Language Search** — "How much on food this month?"

---

## 📁 Data

All data is stored locally in `data/expenses.db` (SQLite). No cloud, no accounts.
