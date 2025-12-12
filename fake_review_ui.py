import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from lime.lime_text import LimeTextExplainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout

# ------------------------
# CONFIG
# ------------------------
MODEL_PATH = "best_model.h5"
MAX_WORDS = 20000
MAX_LEN = 200
EMBED_DIM = 128

# ------------------------
# Helper Functions
# ------------------------


def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http\S+|www\S+|https\S+", " url ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ✅ UPDATED MODEL: swapped LSTM for GRU, with recurrent dropout and added sensible defaults
def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM, input_length=MAX_LEN))

    # Single bidirectional GRU. Use return_sequences=True if you plan to stack more recurrent layers.
    model.add(
        Bidirectional(
            GRU(
                64,
                dropout=0.3,           # input-to-hidden dropout
                recurrent_dropout=0.3  # recurrent dropout
            )
        )
    )

    model.add(Dropout(0.6))    # increased dropout

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.6))    # increased dropout

    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def lime_explain(model, tokenizer, text):
    explainer = LimeTextExplainer(class_names=["Original", "AI Generated"])

    def pred_fn(samples):
        seq = tokenizer.texts_to_sequences(samples)
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        preds = model.predict(padded)
        # preds shape: (n,1) -> flatten to (n,)
        preds = preds.reshape(-1, 1) if preds.ndim == 1 else preds
        return np.hstack([1 - preds, preds])

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=pred_fn,
        num_features=10,
    )
    return exp


# --------- automatic column / label handling ----------


def detect_text_column(df: pd.DataFrame) -> str:
    candidates = ["text_", "review_text", "text", "content", "review"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("No text column detected in dataset")


def detect_label_column(df: pd.DataFrame) -> str:
    candidates = ["label", "review_type", "class"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("No label/target column detected in dataset")


def detect_category_column(df: pd.DataFrame):
    candidates = ["category", "product_category", "duct_categ", "duct_category", "category_name"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_rating_column(df: pd.DataFrame):
    candidates = ["rating", "stars", "score"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_labels(series: pd.Series) -> pd.Series:
    mapping = {
        "cg": 1,
        "ai": 1,
        "computer generated": 1,
        "gpt": 1,
        "gpt-4": 1,
        "gpt4": 1,
        "chatgpt": 1,
        "or": 0,
        "original": 0,
        "human": 0,
    }

    def map_label(x):
        x_str = str(x).strip().lower()
        if x_str.isdigit():  # already numeric
            return int(x_str)
        if x_str in mapping:
            return mapping[x_str]
        # fallback: try to interpret boolean-like values
        if x_str in ("true", "t", "1"):
            return 1
        if x_str in ("false", "f", "0"):
            return 0
        raise ValueError(f"Unknown label value: {x}")

    return series.apply(map_label)


def load_and_standardize_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    text_col = detect_text_column(df)
    label_col = detect_label_column(df)
    cat_col = detect_category_column(df)
    rating_col = detect_rating_column(df)

    out = pd.DataFrame()
    out["text_"] = df[text_col].astype(str).apply(clean_text)
    out["label"] = normalize_labels(df[label_col])

    # category -> if column present, convert to string; else fill with "Unknown"
    if cat_col:
        out["category"] = df[cat_col].astype(str)
    else:
        out["category"] = "Unknown"

    # rating -> present column or NaN
    if rating_col:
        out["rating"] = df[rating_col]
    else:
        out["rating"] = np.nan

    return out


# ------------------------
# UI DESIGN
# ------------------------
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🧠",
    layout="wide",
)

# ---- Session state init ----
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

st.title("🧠 AI-Generated Review Detector")
st.caption("Bi-directional GRU + LIME Explainability")

# multiple dataset uploader
uploaded_files = st.sidebar.file_uploader(
    "📌 Upload one or more CSV datasets",
    type=["csv"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more CSV files to begin.")
    st.stop()

# ------------------------
# Load & combine datasets
# ------------------------
df_list = []
for f in uploaded_files:
    try:
        df_part = load_and_standardize_csv(f)
        df_list.append(df_part)
        st.sidebar.success(f"Loaded: {f.name}")
    except Exception as e:
        st.sidebar.error(f"{f.name}: {e}")

# safety: if nothing loaded, show helpful message and stop
if not df_list:
    st.error(
        "No valid datasets loaded. Make sure your CSVs contain a text column (e.g., 'text', 'review', 'content') and a label column (e.g., 'label', 'class', 'review_type')."
    )
    st.stop()

# concatenate only when we have data (prevents ValueError: No objects to concatenate)
df_all = pd.concat(df_list, ignore_index=True)

st.subheader("📊 Combined Dataset Preview")
st.write(f"Total samples: **{len(df_all)}**")
st.dataframe(df_all.head())

st.write("Label distribution (0 = Original/Human, 1 = AI/CG):")
# guard: if label column missing or all NaN
if "label" in df_all.columns and df_all["label"].notna().any():
    st.bar_chart(df_all["label"].value_counts())
else:
    st.write("No valid label column found in combined dataset.")

# ------------------------
# Optional Filters
# ------------------------
df_filtered = df_all.copy()

st.sidebar.subheader("Filter before training")

if "category" in df_all.columns:
    categories = sorted(df_all["category"].dropna().unique())
    if categories:
        selected_cats = st.sidebar.multiselect("Select Categories", categories, default=categories)
        df_filtered = df_filtered[df_filtered["category"].isin(selected_cats)]

if "rating" in df_all.columns and df_all["rating"].notna().any():
    ratings = sorted(df_all["rating"].dropna().unique())
    selected_ratings = st.sidebar.multiselect("Select Ratings", ratings, default=ratings)
    df_filtered = df_filtered[df_filtered["rating"].isin(selected_ratings)]

if df_filtered.empty:
    st.warning("No records left after applying filters.")
    st.stop()

st.write(f"📂 Filtered dataset size: **{len(df_filtered)} records**")

texts = df_filtered["text_"].tolist()
labels = df_filtered["label"].values

# ------------------------
# Train / Test Split & Tokenization
# ------------------------
# guard: check label distribution to avoid stratify errors
if len(np.unique(labels)) == 1:
    st.error("Only a single class present in the filtered data. Need both classes (0 and 1) to train.")
    st.stop()

X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_txt)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_txt), maxlen=MAX_LEN)
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_txt), maxlen=MAX_LEN)

# ------------------------
# Model buttons
# ------------------------
st.sidebar.divider()

#  ✅ UPDATED TRAINING BLOCK WITH EARLY STOPPING
if st.sidebar.button("🟦 Train New Model"):
    model = build_model()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    with st.spinner("Training model..."):
        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=10,     # we allow more epochs; early stopping prevents overfitting
            batch_size=64,
            verbose=1,
            callbacks=[early_stop]
        )

    model.save(MODEL_PATH)
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    st.success(f"Model trained and saved as {MODEL_PATH}")

if st.sidebar.button("🟩 Load Saved Model"):
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.success("Model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    else:
        st.error("No saved model found. Train first.")

# ------------------------
# Evaluation & Prediction
# ------------------------
model = st.session_state.model

if model is not None:
    tok = st.session_state.tokenizer or tokenizer

    st.subheader("📑 Model Performance")

    try:
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        loss, acc = np.nan, np.nan

    col1, col2 = st.columns(2)
    with col1:
        try:
            st.metric("Accuracy", f"{acc:.4f}")
        except Exception:
            st.metric("Accuracy", "N/A")
    with col2:
        try:
            st.metric("Loss", f"{loss:.4f}")
        except Exception:
            st.metric("Loss", "N/A")

    try:
        preds_raw = model.predict(X_test)
        y_pred = (preds_raw > 0.5).astype(int).reshape(-1)
        st.code(classification_report(y_test, y_pred), language="text")
    except Exception as e:
        st.warning(f"Prediction/Evaluation metrics failed: {e}")

    st.write("Confusion Matrix:")
    try:
        st.write(confusion_matrix(y_test, y_pred))
    except Exception as e:
        st.write(f"Could not compute confusion matrix: {e}")

    st.divider()
    st.subheader("🔍 Test a Review")

    user_text = st.text_area("Enter review text:")

    if user_text:
        cleaned = clean_text(user_text)
        seq = tok.texts_to_sequences([cleaned])
        pad_seq = pad_sequences(seq, maxlen=MAX_LEN)
        try:
            prob = float(model.predict(pad_seq)[0][0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            prob = 0.0

        label = "AI-Generated" if prob >= 0.5 else "Original"
        color = "#ffcccc" if prob >= 0.5 else "#ccffcc"

        st.markdown(
            f"""
            <div style="padding:15px;border-radius:10px;background:{color}">
            <h3 style="margin:0;">{label}</h3>
            <p>Probability: <b>{prob:.4f}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------
        # CLEAN EXPLAINABILITY
        # ----------------
        try:
            st.write("### Explainability (Clean View)")

            exp = lime_explain(model, tok, cleaned)
            weights = exp.as_list()

            ai_words = [w for w, v in weights if v > 0]
            human_words = [w for w, v in weights if v < 0]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔶 Indicates AI-style")
                st.write(", ".join(ai_words[:8]) if ai_words else "None")

            with col2:
                st.markdown("#### 🔷 Indicates Human-style")
                st.write(", ".join(human_words[:8]) if human_words else "None")

            def highlight_text(text, ai_words, human_words):
                ai_set = {a.lower() for a in ai_words}
                human_set = {h.lower() for h in human_words}
                result = []

                for word in text.split():
                    w = word.lower().strip(".,!?")
                    if w in ai_set:
                        result.append(f"<span style='background-color:#ffb347;padding:2px 4px;border-radius:4px;'>{word}</span>")
                    elif w in human_set:
                        result.append(f"<span style='background-color:#6bb4ff;padding:2px 4px;border-radius:4px;'>{word}</span>")
                    else:
                        result.append(word)
                return " ".join(result)

            highlighted = highlight_text(user_text, ai_words, human_words)

            st.markdown("#### Highlighted Sentence Interpretation", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:18px;line-height:1.7;margin-top:10px'>{highlighted}</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"Explainability failed: {e}")

else:
    st.info("Train or load a model to enable evaluation and prediction.")
