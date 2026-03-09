"""
Smart Expense Tracker - Flask Backend
ML-powered personal finance app for Indian users
"""
import os, json, datetime, sqlite3, base64, io, calendar, zoneinfo
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from ml.engine import MLEngine

app = Flask(__name__)
DB_PATH = Path("data/expenses.db")
ml = MLEngine(DB_PATH)

# ── DB SETUP ─────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with get_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            wallet TEXT NOT NULL,
            date TEXT NOT NULL,
            tags TEXT DEFAULT '',
            receipt_path TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS wallets (
            name TEXT PRIMARY KEY,
            balance REAL NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            amount REAL NOT NULL,
            billing_day INTEGER NOT NULL,
            category TEXT DEFAULT 'Subscriptions',
            wallet TEXT DEFAULT 'Bank',
            active INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS budgets (
            month TEXT PRIMARY KEY,
            amount REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS challenges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            target_category TEXT,
            target_amount REAL,
            start_date TEXT,
            end_date TEXT,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS income (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL DEFAULT 'Salary',
            wallet TEXT NOT NULL DEFAULT 'Bank',
            note TEXT DEFAULT '',
            date TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        INSERT OR IGNORE INTO wallets VALUES ('Cash', 5000);
        INSERT OR IGNORE INTO wallets VALUES ('Bank', 20000);
        INSERT OR IGNORE INTO wallets VALUES ('UPI', 10000);
        INSERT OR IGNORE INTO wallets VALUES ('Credit Card', 0);
        INSERT OR IGNORE INTO settings VALUES ('points', '0');
        INSERT OR IGNORE INTO settings VALUES ('onboarded', 'false');
        """)
        # Always ensure current month has a budget set
        month = datetime.date.today().strftime("%Y-%m")
        db.execute("INSERT OR IGNORE INTO budgets VALUES (?, 30000)", (month,))

# ── UTILS ─────────────────────────────────────────────────────────────────────
def setting(key):
    with get_db() as db:
        r = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return r["value"] if r else None

def set_setting(key, val):
    with get_db() as db:
        db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, str(val)))

def current_month():
    return datetime.datetime.now(zoneinfo.ZoneInfo("Asia/Kolkata")).strftime("%Y-%m")

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

def today():
    return datetime.datetime.now(IST).date().isoformat()

def now_ist():
    return datetime.datetime.now(IST).date()

def month_budget():
    with get_db() as db:
        r = db.execute("SELECT amount FROM budgets WHERE month=?", (current_month(),)).fetchone()
        return float(r["amount"]) if r else 30000.0

def month_spent():
    with get_db() as db:
        r = db.execute(
            "SELECT COALESCE(SUM(amount),0) as s FROM expenses WHERE date LIKE ?",
            (current_month() + "%",)
        ).fetchone()
        return float(r["s"])

def month_income():
    with get_db() as db:
        r = db.execute(
            "SELECT COALESCE(SUM(amount),0) as s FROM income WHERE date LIKE ?",
            (current_month() + "%",)
        ).fetchone()
        return float(r["s"])

# ── EXPENSE ROUTES ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/dashboard")
def dashboard():
    spent = month_spent()
    budget = month_budget()
    today_d = now_ist()
    days_in_month = calendar.monthrange(today_d.year, today_d.month)[1]
    days_left = days_in_month - today_d.day
    day_of_month = today_d.day

    with get_db() as db:
        wallets = {r["name"]: r["balance"] for r in db.execute("SELECT * FROM wallets").fetchall()}
        recent = [dict(r) for r in db.execute(
            "SELECT * FROM expenses ORDER BY date DESC, id DESC LIMIT 20"
        ).fetchall()]
        subs = [dict(r) for r in db.execute("SELECT * FROM subscriptions WHERE active=1").fetchall()]
        total_sub = sum(s["amount"] for s in subs)
        income_this_month = month_income()
        recent_income = [dict(r) for r in db.execute(
            "SELECT * FROM income ORDER BY date DESC, id DESC LIMIT 5"
        ).fetchall()]

    total_balance = sum(wallets.values())
    daily_allowed = (budget - spent) / max(days_left, 1)
    avg_daily = spent / max(day_of_month, 1)
    projected = avg_daily * days_in_month
    overspend = projected - budget

    score = ml.habit_score()
    alerts = ml.generate_alerts(spent, budget, days_left, daily_allowed)
    predictions = ml.spending_predictions(spent, budget, days_left, day_of_month, days_in_month)

    return jsonify({
        "balance": total_balance,
        "wallets": wallets,
        "spent": spent,
        "budget": budget,
        "remaining": budget - spent,
        "daily_allowed": daily_allowed,
        "days_left": days_left,
        "avg_daily": avg_daily,
        "projected": projected,
        "overspend": overspend,
        "score": score,
        "alerts": alerts,
        "predictions": predictions,
        "recent": recent,
        "subs": subs,
        "total_sub": total_sub,
        "points": int(setting("points") or 0),
        "income_this_month": income_this_month,
        "net_this_month": income_this_month - spent,
        "recent_income": recent_income,
    })

@app.route("/api/expenses")
def get_expenses():
    month = request.args.get("month", current_month())
    with get_db() as db:
        rows = [dict(r) for r in db.execute(
            "SELECT * FROM expenses WHERE date LIKE ? ORDER BY date DESC, id DESC",
            (month + "%",)
        ).fetchall()]
    return jsonify(rows)

@app.route("/api/expenses", methods=["POST"])
def add_expense():
    d = request.json
    desc = d["description"]
    amount = float(d["amount"])
    wallet = d.get("wallet", "UPI")
    date = d.get("date", today())
    # ML auto-categorize if not provided
    category = d.get("category") or ml.categorize(desc, amount)

    with get_db() as db:
        db.execute(
            "INSERT INTO expenses (description, amount, category, wallet, date) VALUES (?,?,?,?,?)",
            (desc, amount, category, wallet, date)
        )
        db.execute("UPDATE wallets SET balance = balance - ? WHERE name = ?", (amount, wallet))

    pts = int(setting("points") or 0) + 10
    set_setting("points", pts)
    ml.retrain()  # re-train with new data point

    return jsonify({"ok": True, "category": category, "points": pts})

@app.route("/api/expenses/<int:eid>", methods=["DELETE"])
def delete_expense(eid):
    with get_db() as db:
        row = db.execute("SELECT * FROM expenses WHERE id=?", (eid,)).fetchone()
        if row:
            db.execute("UPDATE wallets SET balance = balance + ? WHERE name = ?", (row["amount"], row["wallet"]))
            db.execute("DELETE FROM expenses WHERE id=?", (eid,))
    return jsonify({"ok": True})

@app.route("/api/categorize", methods=["POST"])
def categorize():
    d = request.json
    cat = ml.categorize(d.get("description",""), float(d.get("amount",0)))
    return jsonify({"category": cat})

# ── ML / INSIGHTS ROUTES ──────────────────────────────────────────────────────
@app.route("/api/insights")
def insights():
    month = request.args.get("month", current_month())
    with get_db() as db:
        rows = [dict(r) for r in db.execute(
            "SELECT * FROM expenses WHERE date LIKE ?", (month + "%",)
        ).fetchall()]

    by_cat = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + r["amount"]

    by_wallet = {}
    for r in rows:
        by_wallet[r["wallet"]] = by_wallet.get(r["wallet"], 0) + r["amount"]

    by_day = {}
    for r in rows:
        day = r["date"]
        by_day[day] = by_day.get(day, 0) + r["amount"]

    anomalies = ml.detect_anomalies()
    forecast = ml.forecast_next_30_days()
    cluster_insights = ml.spending_clusters()
    habit_insights = ml.habit_insights()

    return jsonify({
        "by_category": by_cat,
        "by_wallet": by_wallet,
        "by_day": sorted(by_day.items()),
        "anomalies": anomalies,
        "forecast": forecast,
        "clusters": cluster_insights,
        "habit_insights": habit_insights,
        "total": sum(r["amount"] for r in rows),
    })

@app.route("/api/query", methods=["POST"])
def nl_query():
    q = request.json.get("query", "")
    result = ml.natural_language_query(q)
    return jsonify({"answer": result})

@app.route("/api/advice", methods=["POST"])
def get_advice():
    d = request.json
    prompt = d.get("prompt", "Give me financial advice")
    context = ml.build_context()
    answer = ml.ai_advice(prompt, context)
    return jsonify({"answer": answer})

@app.route("/api/receipt", methods=["POST"])
def scan_receipt():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    file = request.files["image"]
    img_bytes = file.read()
    result = ml.scan_receipt(img_bytes, file.content_type)
    return jsonify(result)

# ── WALLET ROUTES ─────────────────────────────────────────────────────────────
@app.route("/api/wallets", methods=["GET"])
def get_wallets():
    with get_db() as db:
        return jsonify({r["name"]: r["balance"] for r in db.execute("SELECT * FROM wallets").fetchall()})

@app.route("/api/wallets", methods=["POST"])
def update_wallet():
    d = request.json
    with get_db() as db:
        db.execute("INSERT OR REPLACE INTO wallets VALUES (?,?)", (d["name"], float(d["balance"])))
    return jsonify({"ok": True})

# ── BUDGET ROUTES ─────────────────────────────────────────────────────────────
@app.route("/api/budget", methods=["GET"])
def get_budget():
    return jsonify({"budget": month_budget(), "month": current_month()})

@app.route("/api/budget", methods=["POST"])
def set_budget():
    d = request.json
    with get_db() as db:
        db.execute("INSERT OR REPLACE INTO budgets VALUES (?,?)",
                   (d.get("month", current_month()), float(d["amount"])))
    return jsonify({"ok": True})

# ── SUBSCRIPTION ROUTES ───────────────────────────────────────────────────────
@app.route("/api/subscriptions", methods=["GET"])
def get_subs():
    with get_db() as db:
        return jsonify([dict(r) for r in db.execute("SELECT * FROM subscriptions WHERE active=1").fetchall()])

@app.route("/api/subscriptions", methods=["POST"])
def add_sub():
    d = request.json
    with get_db() as db:
        db.execute("INSERT INTO subscriptions (name, amount, billing_day, category, wallet) VALUES (?,?,?,?,?)",
                   (d["name"], float(d["amount"]), int(d["billing_day"]), d.get("category","Subscriptions"), d.get("wallet","Bank")))
    return jsonify({"ok": True})

@app.route("/api/subscriptions/<int:sid>", methods=["DELETE"])
def del_sub(sid):
    with get_db() as db:
        db.execute("UPDATE subscriptions SET active=0 WHERE id=?", (sid,))
    return jsonify({"ok": True})

# ── CHALLENGES ────────────────────────────────────────────────────────────────
@app.route("/api/challenges", methods=["GET"])
def get_challenges():
    with get_db() as db:
        return jsonify([dict(r) for r in db.execute("SELECT * FROM challenges").fetchall()])

@app.route("/api/challenges", methods=["POST"])
def add_challenge():
    d = request.json
    with get_db() as db:
        db.execute("INSERT INTO challenges (title, target_category, target_amount, start_date, end_date) VALUES (?,?,?,?,?)",
                   (d["title"], d.get("category",""), float(d.get("target_amount",0)), today(), d.get("end_date","")))
    return jsonify({"ok": True})

@app.route("/api/challenges/<int:cid>/complete", methods=["POST"])
def complete_challenge(cid):
    with get_db() as db:
        db.execute("UPDATE challenges SET status='completed' WHERE id=?", (cid,))
    pts = int(setting("points") or 0) + 150
    set_setting("points", pts)
    return jsonify({"ok": True, "points": pts})

# ── INCOME ROUTES ─────────────────────────────────────────────────────────────
@app.route("/api/income", methods=["GET"])
def get_income():
    month = request.args.get("month", "all")
    with get_db() as db:
        if month == "all":
            rows = [dict(r) for r in db.execute("SELECT * FROM income ORDER BY date DESC, id DESC").fetchall()]
        else:
            rows = [dict(r) for r in db.execute(
                "SELECT * FROM income WHERE date LIKE ? ORDER BY date DESC, id DESC",
                (month + "%",)
            ).fetchall()]
    return jsonify(rows)

@app.route("/api/income", methods=["POST"])
def add_income():
    d = request.json
    source = d.get("source", "").strip()
    amount = float(d.get("amount", 0))
    category = d.get("category", "Salary")
    wallet = d.get("wallet", "Bank")
    note = d.get("note", "")
    date = d.get("date", today())
    if not source or amount <= 0:
        return jsonify({"ok": False, "error": "Source and amount required"}), 400
    with get_db() as db:
        db.execute(
            "INSERT INTO income (source, amount, category, wallet, note, date) VALUES (?,?,?,?,?,?)",
            (source, amount, category, wallet, note, date)
        )
        db.execute("UPDATE wallets SET balance = balance + ? WHERE name = ?", (amount, wallet))
    return jsonify({"ok": True})

@app.route("/api/income/<int:iid>", methods=["DELETE"])
def delete_income(iid):
    with get_db() as db:
        row = db.execute("SELECT * FROM income WHERE id=?", (iid,)).fetchone()
        if row:
            db.execute("UPDATE wallets SET balance = balance - ? WHERE name = ?", (row["amount"], row["wallet"]))
            db.execute("DELETE FROM income WHERE id=?", (iid,))
    return jsonify({"ok": True})

@app.route("/api/income/summary")
def income_summary():
    with get_db() as db:
        all_income = [dict(r) for r in db.execute("SELECT * FROM income ORDER BY date DESC").fetchall()]
    by_month = {}
    by_source = {}
    by_category = {}
    for r in all_income:
        m = r["date"][:7]
        by_month[m] = by_month.get(m, 0) + r["amount"]
        by_source[r["source"]] = by_source.get(r["source"], 0) + r["amount"]
        by_category[r["category"]] = by_category.get(r["category"], 0) + r["amount"]
    total = sum(r["amount"] for r in all_income)
    return jsonify({
        "total": total,
        "by_month": sorted(by_month.items()),
        "by_source": sorted(by_source.items(), key=lambda x: -x[1]),
        "by_category": sorted(by_category.items(), key=lambda x: -x[1]),
        "all": all_income,
    })

@app.route("/api/export/csv")
def export_csv():
    with get_db() as db:
        rows = db.execute("SELECT * FROM expenses ORDER BY date DESC").fetchall()
    lines = ["Date,Description,Amount,Category,Wallet"]
    for r in rows:
        lines.append(f'{r["date"]},"{r["description"]}",{r["amount"]},{r["category"]},{r["wallet"]}')
    csv = "\n".join(lines)
    return app.response_class(csv, mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=expenses.csv"})

if __name__ == "__main__":
    init_db()
    print("\n" + "="*55)
    print("  💰  Smart Expense Tracker  (ML-Powered)")
    print("="*55)
    print("  → Open http://127.0.0.1:5000 in your browser")
    print("  → Press Ctrl+C to stop")
    print("="*55 + "\n")
    app.run(debug=False, port=5000)
