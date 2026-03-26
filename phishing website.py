#!/usr/bin/env python3
"""
Single-file phishing website detector with Flask web interface.
"""

import re
import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, render_template_string

# ---------------------------
# 1. Feature Extraction
# ---------------------------
def extract_features(url):
    features = {}
    parsed = urlparse(url)
    hostname = parsed.netloc
    path = parsed.path

    features['url_length'] = len(url)
    features['hostname_length'] = len(hostname)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_slashes'] = url.count('/')
    features['num_question_marks'] = url.count('?')
    features['num_equals'] = url.count('=')
    features['num_and'] = url.count('&')

    special_chars = set('!@#$%^&*()_+{}[]|\\;:\'",.<>?/')
    features['num_special_chars'] = sum(1 for c in url if c in special_chars)

    ip_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
    features['has_ip'] = 1 if ip_pattern.search(hostname) else 0

    features['has_at'] = 1 if '@' in url else 0
    features['is_https'] = 1 if parsed.scheme == 'https' else 0

    suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'club', 'loan']
    tld = hostname.split('.')[-1] if hostname else ''
    features['suspicious_tld'] = 1 if tld in suspicious_tlds else 0

    parts = hostname.split('.')
    features['num_subdomains'] = len(parts) - 2 if len(parts) > 2 else 0

    features['has_double_slash'] = 1 if '//' in path else 0
    return features

# ---------------------------
# 2. Sample Data (replace for real use)
# ---------------------------
sample_data = [
    ("https://www.google.com", 0),
    ("https://github.com", 0),
    ("https://stackoverflow.com/questions/ask", 0),
    ("https://www.python.org/downloads/", 0),
    ("https://www.wikipedia.org", 0),
    ("https://www.amazon.com", 0),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", 0),
    ("http://secure-paypal.com.login.verify.account.com", 1),
    ("https://www.faceb00k.com/login.php", 1),
    ("http://appleid.apple.com.verify-account.xyz", 1),
    ("https://paypal.com.security-update.ga", 1),
    ("http://192.168.1.100/paypal/login", 1),
    ("https://www.google.com.secure-login.tk", 1),
    ("http://verify-account.instagram.xyz", 1),
]

X_list = [extract_features(url) for url, _ in sample_data]
y_list = [label for _, label in sample_data]

X = pd.DataFrame(X_list)
y = pd.Series(y_list)

# ---------------------------
# 3. Train Model
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model trained on sample dataset.")
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification report (test set):")
print(classification_report(y_test, y_pred))

# ---------------------------
# 4. Flask Web Interface
# ---------------------------
app = Flask(__name__)

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head><title>Phishing URL Detector</title>
<style>
body { font-family: Arial; margin:40px; text-align:center; }
input[type="text"] { width:300px; padding:8px; font-size:16px; }
input[type="submit"] { padding:8px 16px; font-size:16px; cursor:pointer; }
</style>
</head>
<body>
<h1>Phishing URL Detector</h1>
<form action="/predict" method="post">
    <input type="text" name="url" placeholder="Enter a URL (e.g., https://www.google.com)" required>
    <input type="submit" value="Check">
</form>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html>
<head><title>Phishing URL Detector - Result</title>
<style>
body { font-family: Arial; margin:40px; text-align:center; }
.result-box { margin:20px auto; padding:20px; border:1px solid #ccc; width:60%; background:#f9f9f9; }
.phishing { color:red; font-weight:bold; }
.legitimate { color:green; font-weight:bold; }
.back-link { margin-top:20px; }
</style>
</head>
<body>
<h1>Result for:</h1>
<div class="result-box">
    <p><strong>{{ url }}</strong></p>
    <p>Classification: <span class="{% if result == 'Phishing' %}phishing{% else %}legitimate{% endif %}">{{ result }}</span></p>
    <p>Confidence: {{ confidence }}%</p>
</div>
<div class="back-link"><a href="/">Check another URL</a></div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(INDEX_HTML)

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    features = extract_features(url)
    features_df = pd.DataFrame([features])[X.columns]
    pred = model.predict(features_df)[0]
    proba = model.predict_proba(features_df)[0].max()

    result = "Phishing" if pred == 1 else "Legitimate"
    confidence = round(proba * 100, 2)
    return render_template_string(RESULT_HTML, url=url, result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)