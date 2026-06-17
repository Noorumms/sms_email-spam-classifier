# 📩 SMS / Email Spam Classifier

> End-to-end NLP pipeline that detects spam messages with **97%+ accuracy** and **100% precision** — built, tuned, and deployed as a live web app.

🔗 **[Live Demo →](https://your-app.streamlit.app)**

---

## What It Does

Paste any SMS or email text and get an instant spam/ham prediction. The model was trained on 5,500+ real messages and optimized specifically for **precision** — meaning it almost never flags a legitimate message as spam.

---

## How It Works

```
Raw Text → Preprocessing → TF-IDF Vectorization → Multinomial Naive Bayes → Prediction
```

**Preprocessing pipeline:**
- Lowercasing → Tokenization → Removing stopwords & punctuation → Porter Stemming

**Why Multinomial Naive Bayes?**  
Benchmarked 11 classifiers (SVM, Random Forest, XGBoost, LR, KNN, etc.). MNB consistently delivered the best precision on this imbalanced dataset. Precision was prioritized over accuracy because a false positive (ham marked as spam) is far more costly than a false negative.

---

## Results

| Model | Accuracy | Precision |
|---|---|---|
| Multinomial NB ✅ | 97.1% | 100% |
| SVM (sigmoid) | 97.6% | 97.4% |
| Extra Trees | 97.5% | 97.5% |
| Logistic Regression | 95.6% | 97.1% |
| Random Forest | 97.8% | 98.1% |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat-square)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=flat-square)

- **ML:** scikit-learn, NLTK, NumPy, pandas
- **EDA:** Matplotlib, Seaborn, WordCloud
- **Deployment:** Streamlit Community Cloud

---

## Project Structure

```
sms_email-spam-classifier/
├── app.py                  # Streamlit frontend
├── model.pkl               # Trained MNB model
├── vectorizer.pkl          # Fitted TF-IDF vectorizer
├── SMS-spam-Detection.ipynb   # Full EDA + training notebook
└── requirements.txt
```

---

## Run Locally

**1. Clone the repo**
```bash
git https://github.com/Noorumms/sms_email-spam-classifier.git
cd sms_email-spam-classifier
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## Requirements

```
streamlit
scikit-learn
nltk
numpy
pandas
```

---

## Key Learnings

- Imbalanced datasets (87% ham / 13% spam) require optimizing for **precision**, not just accuracy
- TF-IDF with `max_features=3000` outperformed raw Bag-of-Words and scaled features
- Ensemble methods (Voting, Stacking) didn't beat a well-tuned MNB on this task — sometimes simpler is better

---
