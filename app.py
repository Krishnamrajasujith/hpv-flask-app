from flask import Flask, request, redirect, url_for, session
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
import joblib
import os
import sqlite3

app = Flask(__name__)
app.secret_key = "hpv_super_secret_key"

DATABASE = "users.db"
MODEL_PATH = "rf_smote_model.pkl"

# ================= CREATE DATABASE =================
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ================= MODEL SETTINGS =================
TOP_GENES = [
    'CXCL8','CXCL10','CCL20','IFIT1','MX1','OAS1','ISG15',
    'RSAD2','IFIT2','IFIT3','BST2','XAF1','CXCL9','CXCL11',
    'IRF7','STAT1','IRF1','GBP1','HLA-F','TAP1','PSMB9',
    'SAMD9','USP18','DDX58','IFIH1','MDA5','CXCL2','CXCR4'
]

class VirusDetector:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.is_trained = True

    def train_model(self, file):
        df = pd.read_csv(file)
        required_columns = TOP_GENES + ['HPV_Status']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Missing column: {col}")

        X = df[TOP_GENES]
        y = df['HPV_Status']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(X_train_smote, y_train_smote)

        joblib.dump(self.model, MODEL_PATH)
        self.is_trained = True

        auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:,1])
        return auc

    def predict(self, file):
        df = pd.read_csv(file)
        for col in TOP_GENES:
            if col not in df.columns:
                raise KeyError(f"Missing column: {col}")

        row = df.iloc[0]
        sample = [row[gene] for gene in TOP_GENES]
        prob = self.model.predict_proba([sample])[0][1]
        return prob

detector = VirusDetector()

# ================= HOME =================
@app.route('/')
def home():
    return """
    <h2>🦠 HPV16 Detection System</h2>
    <a href='/register'>Register</a><br><br>
    <a href='/login'>Login</a>
    """

# ================= REGISTER =================
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = sqlite3.connect(DATABASE)
            c = conn.cursor()
            c.execute("INSERT INTO users (username,password) VALUES (?,?)",
                      (username,password))
            conn.commit()
            conn.close()
            return "<h3 style='color:green;'>Registration Successful ✅</h3><a href='/login'>Go to Login</a>"
        except:
            return "<h3 style='color:red;'>Username already exists ❌</h3><a href='/register'>Try Again</a>"
    return """
    <h2>📝 Register</h2>
    <form method='post'>
        Username:<br>
        <input type='text' name='username' required><br><br>
        Password:<br>
        <input type='password' name='password' required><br><br>
        <button type='submit'>Register</button>
    </form>
    """

# ================= LOGIN =================
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?",
                  (username,password))
        user = c.fetchone()
        conn.close()

        if user:
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return "<h3 style='color:red;'>Invalid Login ❌</h3><a href='/login'>Try Again</a>"
    return """
    <h2>🔐 Login</h2>
    <form method='post'>
        Username:<br>
        <input type='text' name='username' required><br><br>
        Password:<br>
        <input type='password' name='password' required><br><br>
        <button type='submit'>Login</button>
    </form>
    """

# ================= DASHBOARD =================
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return f"""
    <h2>Welcome {session['user']} 👋</h2>
    <h3>🦠 HPV16 Detection Dashboard</h3>

    <h4>Train Model</h4>
    <form action='/train' method='post' enctype='multipart/form-data'>
        <input type='file' name='file' required>
        <button type='submit'>Train Model</button>
    </form>

    <hr>

    <h4>Predict Patient</h4>
    <form action='/predict' method='post' enctype='multipart/form-data'>
        <input type='file' name='file' required>
        <button type='submit'>Check HPV Status</button>
    </form>

    <br><br>
    <a href='/logout'>Logout</a>
    """

# ================= TRAIN =================
@app.route('/train', methods=['POST'])
def train():
    if 'user' not in session:
        return redirect(url_for('login'))
    try:
        file = request.files['file']
        auc = detector.train_model(file)
        return f"""
        <h3 style='color:green;'>Model Trained Successfully ✅</h3>
        <h4>AUC Score: {round(auc,3)}</h4>
        <a href='/dashboard'>Back</a>
        """
    except KeyError as e:
        return f"""
        <h3 style='color:red;'>Training Failed ❌</h3>
        <p>{str(e)}</p>
        <a href='/dashboard'>Back</a>
        """
    except Exception as e:
        return f"""
        <h3 style='color:red;'>Unexpected Error ❌</h3>
        <p>{str(e)}</p>
        <a href='/dashboard'>Back</a>
        """

# ================= PREDICT =================
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
    try:
        if not detector.is_trained:
            return "<h3>Model not trained yet.</h3><a href='/dashboard'>Back</a>"

        file = request.files['file']
        prob = detector.predict(file)
        status = "HPV16 Positive" if prob > 0.5 else "HPV16 Negative"
        color = "red" if prob > 0.5 else "green"
        return f"""
        <h2>Prediction Result</h2>
        <h3 style='color:{color}'>{status}</h3>
        <h4>Confidence: {round(prob*100,2)}%</h4>
        <a href='/dashboard'>Back</a>
        """
    except KeyError as e:
        return f"""
        <h3 style='color:red;'>Prediction Failed ❌</h3>
        <p>{str(e)}</p>
        <a href='/dashboard'>Back</a>
        """

# ================= LOGOUT =================
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

# ================= RUN SERVER =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # required for cloud deployment
    app.run(host="0.0.0.0", port=port, debug=True)