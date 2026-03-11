"""
ML Engine for Smart Expense Tracker
Uses: scikit-learn, numpy, anthropic API
Models:
  1. Naive Bayes / SVM - auto expense categorization
  2. Linear Regression - spending forecast
  3. Isolation Forest - anomaly detection
  4. K-Means - spending cluster analysis
  5. Habit scoring via rule-based + ML features
  6. Anthropic Claude API - NL queries & AI advice
"""
import os, json, re, sqlite3, datetime, base64
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── CATEGORY CONFIG ───────────────────────────────────────────────────────────
CATEGORIES = [
    "Food", "Transport", "Shopping", "Entertainment",
    "Health", "Utilities", "Rent", "Subscriptions", "Other"
]

# Seed training data for the categorizer (keyword → category)
SEED_DATA = [
    # Food
    ("swiggy order", 250, "Food"), ("zomato dinner", 400, "Food"),
    ("lunch canteen", 80, "Food"), ("breakfast cafe", 120, "Food"),
    ("groceries dmart", 1200, "Food"), ("chai tapri", 20, "Food"),
    ("restaurant dinner", 800, "Food"), ("pizza dominos", 450, "Food"),
    ("ice cream", 100, "Food"), ("biscuits snacks", 50, "Food"),
    ("milk packet", 60, "Food"), ("vegetables sabzi", 200, "Food"),
    ("mcd burger", 300, "Food"), ("biryani order", 350, "Food"),
    # Transport
    ("uber cab", 200, "Transport"), ("ola ride", 150, "Transport"),
    ("metro card recharge", 500, "Transport"), ("bus ticket", 30, "Transport"),
    ("petrol fuel", 1000, "Transport"), ("auto rickshaw", 80, "Transport"),
    ("rapido bike", 50, "Transport"), ("train ticket irctc", 600, "Transport"),
    ("flight ticket", 4000, "Transport"), ("toll tax", 100, "Transport"),
    # Shopping
    ("amazon purchase", 1500, "Shopping"), ("flipkart order", 2000, "Shopping"),
    ("myntra clothes", 800, "Shopping"), ("ajio fashion", 1200, "Shopping"),
    ("shoes footwear", 1500, "Shopping"), ("tshirt shirt", 700, "Shopping"),
    ("decathlon sports", 1800, "Shopping"), ("book flipkart", 400, "Shopping"),
    ("electronics headphones", 2500, "Shopping"), ("watches", 3000, "Shopping"),
    # Entertainment
    ("netflix subscription", 649, "Entertainment"), ("prime video", 299, "Entertainment"),
    ("movie pvr inox", 350, "Entertainment"), ("concert event", 1200, "Entertainment"),
    ("gaming steam", 500, "Entertainment"), ("spotify music", 119, "Entertainment"),
    ("youtube premium", 129, "Entertainment"), ("bookmyshow", 450, "Entertainment"),
    # Health
    ("doctor consultation", 500, "Health"), ("pharmacy medicine", 300, "Health"),
    ("gym membership", 1500, "Health"), ("hospital", 2000, "Health"),
    ("health checkup", 1000, "Health"), ("protein supplement", 1200, "Health"),
    ("yoga class", 800, "Health"), ("dentist", 700, "Health"),
    # Utilities
    ("electricity bill", 1500, "Utilities"), ("water bill", 200, "Utilities"),
    ("gas cylinder", 900, "Utilities"), ("internet wifi bill", 700, "Utilities"),
    ("mobile recharge", 239, "Utilities"), ("dth recharge", 400, "Utilities"),
    # Rent
    ("rent payment", 15000, "Rent"), ("pg accommodation", 8000, "Rent"),
    ("house rent", 12000, "Rent"), ("flat rent deposit", 20000, "Rent"),
    # Subscriptions
    ("hotstar subscription", 299, "Subscriptions"), ("jio fiber", 999, "Subscriptions"),
    ("icloud storage", 75, "Subscriptions"), ("linkedin premium", 2000, "Subscriptions"),
    ("coursera annual", 3000, "Subscriptions"),
    # Other
    ("atm withdrawal", 2000, "Other"), ("gift friend", 500, "Other"),
    ("donation temple", 100, "Other"), ("laundry", 300, "Other"),
    ("stationery office", 200, "Other"), ("haircut salon", 250, "Other"),
]

class MLEngine:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._clf = None
        self._vec = None
        self._iso = None  # anomaly detector
        self._kmeans = None  # spending clusters
        self._trained = False
        self._train_categorizer()

    def _get_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _all_expenses(self):
        try:
            with self._get_db() as db:
                return [dict(r) for r in db.execute("SELECT * FROM expenses ORDER BY date").fetchall()]
        except:
            return []

    # ── 1. CATEGORIZER (Naive Bayes) ──────────────────────────────────────────
    def _features(self, description: str, amount: float):
        """Text + amount features for categorization"""
        text = description.lower()
        # Amount buckets
        amt_feat = [
            1 if amount < 100 else 0,
            1 if 100 <= amount < 500 else 0,
            1 if 500 <= amount < 2000 else 0,
            1 if 2000 <= amount < 10000 else 0,
            1 if amount >= 10000 else 0,
        ]
        return text, amt_feat

    def _train_categorizer(self):
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import ComplementNB
            from sklearn.preprocessing import LabelEncoder

            # Combine seed + real data
            real = self._all_expenses()
            all_data = list(SEED_DATA) + [(r["description"], r["amount"], r["category"]) for r in real]

            texts = [d[0].lower() for d in all_data]
            labels = [d[2] for d in all_data]

            self._vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, analyzer="word", sublinear_tf=True)
            from sklearn.naive_bayes import MultinomialNB
            self._clf = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), analyzer="word", sublinear_tf=True)),
                ("clf", MultinomialNB(alpha=0.3)),
            ])
            self._clf.fit(texts, labels)
            self._trained = True
        except Exception as e:
            self._trained = False

    def retrain(self):
        """Re-train with all available data including newly added"""
        self._train_categorizer()
        self._train_anomaly_detector()
        self._train_clusters()

    def categorize(self, description: str, amount: float = 0) -> str:
        """Smart categorization via Llama — falls back to ML/rules if Llama unavailable"""
        try:
            import requests
            prompt = f"""You are a categorizer for an Indian personal expense tracker.
Given this expense description: "{description}" (amount: ₹{amount})

Reply with ONLY one word from this exact list:
Food, Transport, Shopping, Entertainment, Health, Utilities, Rent, Subscriptions, Other

Examples:
- "petrol for bike" → Transport
- "filled up activa" → Transport  
- "fuel bunk" → Transport
- "VIT canteen milkshake" → Food
- "chicken dinner" → Food
- "blinkit snacks" → Food
- "Netflix" → Subscriptions
- "gym fees" → Health
- "room rent" → Rent
- "electricity bill" → Utilities

Reply with just the category word, nothing else."""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.1", "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.1, "num_predict": 10}},
                timeout=8
            )
            result = response.json().get("response", "").strip()
            # Clean and validate
            for cat in CATEGORIES:
                if cat.lower() in result.lower():
                    return cat
            # fallback if Llama gives unexpected answer
            return self._ml_categorize(description)
        except:
            return self._ml_categorize(description)

    def _ml_categorize(self, description: str) -> str:
        """Fallback: trained Naive Bayes or rule-based"""
        if self._trained:
            try:
                return self._clf.predict([description.lower()])[0]
            except:
                pass
        return self._rule_based_category(description)

    def _rule_based_category(self, desc: str) -> str:
        d = desc.lower()
        if any(w in d for w in ["food","eat","lunch","dinner","break","restaur","cafe","swiggy","zomato","pizza","burger","chai","milk","rice","dal","sabzi","grocer","snack"]): return "Food"
        if any(w in d for w in ["uber","ola","metro","bus","auto","petrol","fuel","train","flight","cab","rapido","toll"]): return "Transport"
        if any(w in d for w in ["amazon","flipkart","shop","cloth","shoes","myntra","ajio","book","electronic"]): return "Shopping"
        if any(w in d for w in ["movie","netflix","prime","spotify","game","concert","pvr","inox","bookmyshow"]): return "Entertainment"
        if any(w in d for w in ["doctor","medic","pharma","hospital","gym","health","dentist","protein","yoga"]): return "Health"
        if any(w in d for w in ["electric","water","gas","internet","wifi","bill","mobile","recharge","dth"]): return "Utilities"
        if any(w in d for w in ["rent","pg","house","flat","landlord","accommo"]): return "Rent"
        if any(w in d for w in ["subscri","monthly","annual","premium","jio","icloud"]): return "Subscriptions"
        return "Other"

    # ── 2. ANOMALY DETECTOR (Isolation Forest) ────────────────────────────────
    def _train_anomaly_detector(self):
        try:
            from sklearn.ensemble import IsolationForest
            expenses = self._all_expenses()
            if len(expenses) < 10:
                self._iso = None
                return
            amounts = np.array([[e["amount"]] for e in expenses])
            self._iso = IsolationForest(contamination=0.1, random_state=42)
            self._iso.fit(amounts)
        except:
            self._iso = None

    def detect_anomalies(self):
        """Detect unusually large or strange expenses"""
        try:
            from sklearn.ensemble import IsolationForest
            expenses = self._all_expenses()
            if len(expenses) < 8:
                return []

            # Group by category, find statistical outliers
            by_cat = defaultdict(list)
            for e in expenses:
                by_cat[e["category"]].append(e)

            anomalies = []
            for cat, exps in by_cat.items():
                if len(exps) < 3:
                    continue
                amounts = [e["amount"] for e in exps]
                mean = np.mean(amounts)
                std = np.std(amounts)
                for e in exps:
                    if std > 0 and (e["amount"] - mean) / std > 2.0:
                        anomalies.append({
                            "id": e["id"],
                            "description": e["description"],
                            "amount": e["amount"],
                            "category": e["category"],
                            "date": e["date"],
                            "z_score": round((e["amount"] - mean) / std, 2),
                            "avg_for_category": round(mean, 2),
                            "message": f"This {cat} expense is {round(e['amount']/mean, 1)}x your average for this category"
                        })
            return sorted(anomalies, key=lambda x: x["z_score"], reverse=True)[:5]
        except:
            return []

    # ── 3. FORECASTING (Linear Regression on time-series) ────────────────────
    def forecast_next_30_days(self):
        """Predict daily spend for next 30 days using linear regression"""
        try:
            from sklearn.linear_model import LinearRegression
            expenses = self._all_expenses()
            if len(expenses) < 5:
                return {"labels": [], "values": [], "message": "Need more data for forecast"}

            # Aggregate by day
            by_day = defaultdict(float)
            for e in expenses:
                by_day[e["date"]] += e["amount"]

            if len(by_day) < 5:
                return {"labels": [], "values": [], "message": "Need more daily data"}

            sorted_days = sorted(by_day.items())
            base = datetime.date.fromisoformat(sorted_days[0][0])
            X = np.array([(datetime.date.fromisoformat(d) - base).days for d, _ in sorted_days]).reshape(-1,1)
            y = np.array([v for _, v in sorted_days])

            reg = LinearRegression()
            reg.fit(X, y)

            # Predict next 30 days
            today = datetime.date.today()
            future_labels = []
            future_vals = []
            for i in range(1, 31):
                fd = today + datetime.timedelta(days=i)
                days_from_base = (fd - base).days
                pred = max(0, reg.predict([[days_from_base]])[0])
                future_labels.append(fd.isoformat())
                future_vals.append(round(float(pred), 2))

            trend = "increasing" if reg.coef_[0] > 10 else "decreasing" if reg.coef_[0] < -10 else "stable"
            return {
                "labels": future_labels,
                "values": future_vals,
                "trend": trend,
                "message": f"Your spending trend is {trend}. Projected avg: ₹{round(np.mean(future_vals))}/day"
            }
        except Exception as e:
            return {"labels": [], "values": [], "message": str(e)}

    # ── 4. K-MEANS SPENDING CLUSTERS ─────────────────────────────────────────
    def _train_clusters(self):
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            expenses = self._all_expenses()
            if len(expenses) < 10:
                self._kmeans = None
                return
            X = np.array([[e["amount"], CATEGORIES.index(e["category"]) if e["category"] in CATEGORIES else 8] for e in expenses])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            k = min(4, len(expenses) // 3)
            self._kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            self._kmeans.fit(X_scaled)
        except:
            self._kmeans = None

    def spending_clusters(self):
        """Identify spending behavior clusters"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            expenses = self._all_expenses()
            if len(expenses) < 10:
                return {"message": "Need at least 10 expenses for cluster analysis", "clusters": []}

            X = np.array([[e["amount"]] for e in expenses])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            k = min(3, len(expenses) // 4)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)

            cluster_info = []
            for i in range(k):
                idxs = [j for j, l in enumerate(labels) if l == i]
                cluster_exps = [expenses[j] for j in idxs]
                avg_amt = np.mean([e["amount"] for e in cluster_exps])
                cats = [e["category"] for e in cluster_exps]
                dominant_cat = max(set(cats), key=cats.count)
                label = "Small daily spends" if avg_amt < 300 else "Medium purchases" if avg_amt < 2000 else "Large expenses"
                cluster_info.append({
                    "label": label,
                    "count": len(cluster_exps),
                    "avg_amount": round(float(avg_amt), 2),
                    "dominant_category": dominant_cat,
                    "examples": [e["description"] for e in cluster_exps[:3]],
                })
            return {"clusters": sorted(cluster_info, key=lambda x: x["avg_amount"])}
        except Exception as e:
            return {"clusters": [], "message": str(e)}

    # ── 5. HABIT SCORE ────────────────────────────────────────────────────────
    def habit_score(self) -> int:
        """ML-informed habit score 0-100"""
        try:
            expenses = self._all_expenses()
            if not expenses:
                return 50

            today = datetime.date.today()
            this_month = [e for e in expenses if e["date"].startswith(today.strftime("%Y-%m"))]
            spent = sum(e["amount"] for e in this_month)

            # Get budget
            with self._get_db() as db:
                r = db.execute("SELECT amount FROM budgets WHERE month=?", (today.strftime("%Y-%m"),)).fetchone()
                budget = float(r["amount"]) if r else 30000.0

            score = 100
            ratio = spent / budget if budget > 0 else 1

            # Budget usage penalty
            if ratio > 1.0: score -= 40
            elif ratio > 0.9: score -= 25
            elif ratio > 0.75: score -= 15
            elif ratio > 0.6: score -= 5

            # Consistency bonus (logging every day)
            days_active = len(set(e["date"] for e in this_month))
            days_elapsed = today.day
            consistency = days_active / max(days_elapsed, 1)
            score += int(consistency * 15)

            # Category diversity (spending across multiple categories is healthy)
            cats = set(e["category"] for e in this_month)
            if len(cats) >= 5: score += 5
            elif len(cats) <= 2: score -= 5

            # Anomaly penalty
            anomalies = self.detect_anomalies()
            score -= len(anomalies) * 3

            # Logging streak bonus
            recent_7 = [e for e in expenses if (today - datetime.date.fromisoformat(e["date"])).days <= 7]
            days_logged_7 = len(set(e["date"] for e in recent_7))
            if days_logged_7 >= 6: score += 10
            elif days_logged_7 >= 4: score += 5

            return max(0, min(100, score))
        except:
            return 50

    def habit_insights(self):
        """ML-powered habit analysis text insights"""
        try:
            expenses = self._all_expenses()
            if len(expenses) < 5:
                return ["Add more expenses to unlock habit insights!"]

            insights = []
            today = datetime.date.today()
            this_month = [e for e in expenses if e["date"].startswith(today.strftime("%Y-%m"))]
            last_month_str = (today.replace(day=1) - datetime.timedelta(days=1)).strftime("%Y-%m")
            last_month = [e for e in expenses if e["date"].startswith(last_month_str)]

            # Month-over-month comparison
            if this_month and last_month:
                curr_total = sum(e["amount"] for e in this_month)
                last_total = sum(e["amount"] for e in last_month)
                if last_total > 0:
                    change_pct = ((curr_total - last_total) / last_total) * 100
                    if change_pct > 20:
                        insights.append(f"📈 You're spending {change_pct:.0f}% more than last month. Consider cutting back.")
                    elif change_pct < -15:
                        insights.append(f"📉 Great job! You've reduced spending by {abs(change_pct):.0f}% vs last month.")

            # Peak spending day analysis
            by_weekday = defaultdict(float)
            for e in expenses[-60:]:
                wd = datetime.date.fromisoformat(e["date"]).weekday()
                by_weekday[wd] += e["amount"]
            if by_weekday:
                peak_day = max(by_weekday, key=by_weekday.get)
                days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                insights.append(f"📅 You spend the most on {days[peak_day]}s. Plan ahead!")

            # Category trend
            if this_month:
                by_cat = defaultdict(float)
                for e in this_month:
                    by_cat[e["category"]] += e["amount"]
                top_cat = max(by_cat, key=by_cat.get)
                top_amt = by_cat[top_cat]
                total = sum(by_cat.values())
                pct = (top_amt / total * 100) if total > 0 else 0
                if pct > 40:
                    insights.append(f"🎯 {top_cat} takes up {pct:.0f}% of your spending this month — your biggest category.")

            # Dining frequency
            food_exps = [e for e in expenses[-30:] if e["category"] == "Food"]
            if len(food_exps) > 20:
                insights.append(f"🍔 You've eaten out/ordered {len(food_exps)} times in 30 days. Cooking more can save ₹{len(food_exps)*150:,}+")

            # Subscription audit
            with self._get_db() as db:
                subs = db.execute("SELECT SUM(amount) as t FROM subscriptions WHERE active=1").fetchone()
                if subs and subs["t"] and subs["t"] > 2000:
                    insights.append(f"📱 You pay ₹{subs['t']:,.0f}/month on subscriptions. Review unused ones!")

            if not insights:
                insights.append("Keep logging expenses to unlock personalized insights!")

            return insights[:5]
        except Exception as e:
            return [f"Insights unavailable: {str(e)}"]

    # ── 6. ALERTS & PREDICTIONS ───────────────────────────────────────────────
    def generate_alerts(self, spent, budget, days_left, daily_allowed):
        alerts = []
        ratio = spent / budget if budget > 0 else 0
        if ratio >= 1.0:
            alerts.append({"type": "danger", "msg": f"🚨 Budget exceeded! You've spent ₹{spent:,.0f} of ₹{budget:,.0f}"})
        elif ratio >= 0.9:
            alerts.append({"type": "danger", "msg": f"🚨 90% budget used! Only ₹{budget-spent:,.0f} left."})
        elif ratio >= 0.75:
            alerts.append({"type": "warn", "msg": f"⚠️ 75% budget used. ₹{budget-spent:,.0f} for {days_left} days."})
        if daily_allowed < 300 and days_left > 0:
            alerts.append({"type": "warn", "msg": f"📉 Daily budget is only ₹{daily_allowed:,.0f}. Spend carefully!"})
        return alerts

    def spending_predictions(self, spent, budget, days_left, day_of_month, days_in_month):
        avg_daily = spent / max(day_of_month, 1)
        projected = avg_daily * days_in_month
        sub_total = 0
        try:
            with self._get_db() as db:
                r = db.execute("SELECT COALESCE(SUM(amount),0) as t FROM subscriptions WHERE active=1").fetchone()
                sub_total = float(r["t"] or 0)
        except:
            pass
        return {
            "projected_month_spend": round(projected, 2),
            "overspend": round(projected - budget, 2),
            "daily_avg": round(avg_daily, 2),
            "cashflow_after_subs": round((budget - spent) - sub_total, 2),
            "safe_to_spend_today": round(max(0, (budget - spent - sub_total) / max(days_left, 1)), 2),
        }

    # ── 7. NATURAL LANGUAGE QUERY ─────────────────────────────────────────────
    def natural_language_query(self, query: str) -> str:
        """Parse NL query and return answer from DB"""
        q = query.lower()
        expenses = self._all_expenses()
        today = datetime.date.today()

        # Time window
        if "today" in q:
            filtered = [e for e in expenses if e["date"] == today.isoformat()]
            window = "today"
        elif "this week" in q or "last 7 days" in q:
            cutoff = (today - datetime.timedelta(days=7)).isoformat()
            filtered = [e for e in expenses if e["date"] >= cutoff]
            window = "this week"
        elif "last month" in q:
            lm = (today.replace(day=1) - datetime.timedelta(days=1)).strftime("%Y-%m")
            filtered = [e for e in expenses if e["date"].startswith(lm)]
            window = "last month"
        else:
            filtered = [e for e in expenses if e["date"].startswith(today.strftime("%Y-%m"))]
            window = "this month"

        # Category filter
        cat_found = None
        for cat in CATEGORIES:
            if cat.lower() in q:
                cat_found = cat
                break

        if cat_found:
            filtered = [e for e in filtered if e["category"] == cat_found]

        total = sum(e["amount"] for e in filtered)
        count = len(filtered)

        if not filtered:
            return f"No expenses found for '{query}'. Try logging some first!"

        # Query type
        if "biggest" in q or "largest" in q or "most" in q:
            top = sorted(filtered, key=lambda x: x["amount"], reverse=True)[:3]
            lines = [f"  • {e['description']} — ₹{e['amount']:,.0f} ({e['date']})" for e in top]
            return f"Biggest expenses {window}:\n" + "\n".join(lines)

        if "average" in q or "avg" in q:
            avg = total / count if count else 0
            return f"Average expense {window}{' on ' + cat_found if cat_found else ''}: ₹{avg:,.0f} (over {count} transactions)"

        if "how many" in q or "count" in q:
            return f"You have {count} expense{'s' if count != 1 else ''} {window}{' in ' + cat_found if cat_found else ''}."

        cat_str = f" on {cat_found}" if cat_found else ""
        return f"Total spent{cat_str} {window}: ₹{total:,.0f} across {count} transaction{'s' if count != 1 else ''}."

    # ── 8. BUILD AI CONTEXT ───────────────────────────────────────────────────
    def build_context(self) -> str:
        try:
            expenses = self._all_expenses()
            today = datetime.date.today()
            this_month = [e for e in expenses if e["date"].startswith(today.strftime("%Y-%m"))]
            spent = sum(e["amount"] for e in this_month)

            by_cat = defaultdict(float)
            for e in this_month:
                by_cat[e["category"]] += e["amount"]

            with self._get_db() as db:
                wallets = {r["name"]: r["balance"] for r in db.execute("SELECT * FROM wallets").fetchall()}
                budget_row = db.execute("SELECT amount FROM budgets WHERE month=?", (today.strftime("%Y-%m"),)).fetchone()
                budget = float(budget_row["amount"]) if budget_row else 30000.0
                subs = [dict(r) for r in db.execute("SELECT * FROM subscriptions WHERE active=1").fetchall()]

            days_in_month = 30
            days_left = days_in_month - today.day
            sub_total = sum(s["amount"] for s in subs)

            context = f"""User Financial Profile (Indian User, currency ₹):
- Today: {today.isoformat()}, Days left in month: {days_left}
- Total Balance: ₹{sum(wallets.values()):,.0f}
- Wallets: {', '.join(f"{k}: ₹{v:,.0f}" for k,v in wallets.items())}
- Monthly Budget: ₹{budget:,.0f}
- Spent this month: ₹{spent:,.0f}
- Remaining budget: ₹{budget-spent:,.0f}
- Category breakdown: {', '.join(f"{k}: ₹{v:,.0f}" for k,v in sorted(by_cat.items(), key=lambda x: -x[1]))}
- Monthly subscriptions: {', '.join(f"{s['name']} ₹{s['amount']}" for s in subs)} (Total: ₹{sub_total:,.0f})
- Habit Score: {self.habit_score()}/100
- Recent expenses (last 5): {', '.join(f"{e['description']} ₹{e['amount']}" for e in expenses[-5:])}"""
            return context
        except:
            return "Financial data unavailable."

    # ── 9. AI ADVICE (Ollama - Local Llama 3.1) ──────────────────────────────
    def ai_advice(self, prompt: str, context: str) -> str:
        try:
            import requests
            full_prompt = f"""{context}

User question: {prompt}

You are a smart, empathetic personal finance advisor for Indian users.
Use ₹ symbol. Be practical, specific, and actionable.
Use bullet points for lists. Keep responses concise (under 200 words).
Reference the user's actual data when giving advice.
Be encouraging but honest about overspending."""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.1",
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 400,
                    }
                },
                timeout=60
            )
            return response.json().get("response", "No response from model.")
        except requests.exceptions.ConnectionError:
            return "Ollama is not running. Open a terminal and run: ollama serve"
        except Exception as e:
            return f"AI advice unavailable: {str(e)}"

    # ── 10. MASCOT REACTIONS (Ollama - Llama 3.1) ─────────────────────────────
    def mascot_reaction(self, action: str, category: str, description: str, context: str) -> str:
        try:
            import requests
            hour = datetime.datetime.now().hour
            time_context = "late night" if 0 <= hour < 5 else "morning" if 5 <= hour < 12 else "afternoon" if 12 <= hour < 17 else "evening"

            prompt = f"""You are Paisa, a wise and sarcastic owl mascot living inside an expense tracker app for an Indian college student.
You have access to their financial data and react to what they do in the app.

{context}
Current time: {time_context}
Action: {action}
Expense description: "{description}"
Category: {category}

Generate ONE short reaction (max 12 words, no quotes) that is:
- Witty, sarcastic or funny but not mean
- Specific to what they just did (reference the actual expense if possible)
- Feels like something a sarcastic desi friend would say
- Occasionally use Indian slang or references (yaar, bhai, arre, etc.)
- If it's food: comment on the eating habits
- If it's girlfriend/date/gift: be playfully teasing
- If they're low on budget: be dramatically concerned
- If income was added: be genuinely hyped
- Never use hashtags or emojis in the text (emojis allowed at end only)

Reply with ONLY the reaction text, nothing else."""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.1",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.9, "num_predict": 40}
                },
                timeout=15
            )
            result = response.json().get("response", "").strip()
            # Clean up any quotes Llama might add
            result = result.strip('"\'').strip()
            return result if result else "Noted. 🦉"
        except:
            return "Logged! 🦉"


    def scan_receipt(self, img_bytes: bytes, media_type: str) -> dict:
        try:
            import anthropic
            client = anthropic.Anthropic()
            b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                        {"type": "text", "text": """Extract from this receipt and return ONLY valid JSON (no markdown):
{"description": "merchant or item name", "amount": total_as_number, "category": "Food|Transport|Shopping|Entertainment|Health|Utilities|Other", "date": "YYYY-MM-DD"}
If date not visible, use today's date. Amount must be a number only."""}
                    ]
                }]
            )
            text = response.content[0].text.strip()
            text = re.sub(r"```json|```", "", text).strip()
            data = json.loads(text)
            return {"ok": True, **data}
        except Exception as e:
            return {"ok": False, "error": str(e)}
