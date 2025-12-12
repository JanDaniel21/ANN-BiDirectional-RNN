import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from lime.lime_text import LimeTextExplainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow import
import tensorflow as tf
import tensorflow.keras.backend as K


# --- CPU Optimization (Ryzen 3600) ---
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(6)
tf.config.optimizer.set_jit(True)

# --- Keras imports ---
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import regularizers



# ------------------------
# CONFIG
# ------------------------
MODEL_PATH = "best_model.h5"
MAX_WORDS = 20000
MAX_LEN = 350
EMBED_DIM = 256

# ------------------------
# Helper Functions
# ------------------------


def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http\\S+|www\\S+|https\\S+", " url ", t)
    t = re.sub(r"[^a-z0-9\\s]", " ", t)
    t = re.sub(r"\\s+", " ", t).strip()
    return t


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -alpha * K.pow(1. - pt, gamma) * K.log(pt)
        return K.mean(loss)
    return focal_loss_fixed

# 
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import AdamW

def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM, input_length=MAX_LEN))

    model.add(Bidirectional(GRU(
        256,
        dropout=0.3,
        recurrent_dropout=0.3,
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        recurrent_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4)
    )))

    model.add(Dropout(0.6))

    model.add(Dense(
        64, activation="relu",
        kernel_regularizer=regularizers.L2(1e-4)
    ))
    model.add(Dropout(0.6))

    model.add(Dense(
        1, activation="sigmoid",
        kernel_regularizer=regularizers.L2(1e-4)
    ))

    optimizer = AdamW(
        learning_rate=3e-4,
        weight_decay=1e-4
    )

    model.compile(
        loss=focal_loss(gamma=2, alpha=0.25),
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model




def lime_explain(model, tokenizer, text):
    explainer = LimeTextExplainer(class_names=["Original", "AI Generated"])

    def pred_fn(samples):
        seq = tokenizer.texts_to_sequences(samples)
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        preds = model.predict(padded)
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
        "Simulated Human": 0,
    }

    def map_label(x):
        x_str = str(x).strip().lower()
        if x_str.isdigit():  # already numeric
            return int(x_str)
        if x_str in mapping:
            return mapping[x_str]
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

    out["category"] = df[cat_col].astype(str) if cat_col else "Unknown"
    out["rating"] = df[rating_col] if rating_col else np.nan

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
st.caption("Bi-directional RNN + LIME Explainability")

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

if not df_list:
    st.error("No valid datasets loaded.")
    st.stop()

df_all = pd.concat(df_list, ignore_index=True)

st.subheader("📊 Combined Dataset Preview")
st.write(f"Total samples: **{len(df_all)}**")
st.dataframe(df_all.head())

st.write("Label distribution (0 = Original/Human, 1 = AI/CG):")
st.bar_chart(df_all["label"].value_counts())

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
        patience=3,
        restore_best_weights=True
    )

    # Apply heavier weight to AI-generated class (1)
    class_weights = {0: 1.0, 1: 1.5}

    with st.spinner("Training model..."):
        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=25,
            batch_size=128,
            verbose=1,
            callbacks=[early_stop],
            class_weight=class_weights
        )

    model.save(MODEL_PATH)
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    st.success("Model trained and saved as best_model.h5")


if st.sidebar.button("🟩 Load Saved Model"):
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.success("Model loaded successfully.")
    else:
        st.error("No saved model found. Train first.")

# ------------------------
# Evaluation & Prediction
# ------------------------
model = st.session_state.model

if model is not None:
    tok = st.session_state.tokenizer or tokenizer

    st.subheader("📑 Model Performance")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{acc:.4f}")
    with col2:
        st.metric("Loss", f"{loss:.4f}")

    y_pred = (model.predict(X_test) > 0.5).astype(int)

    st.code(classification_report(y_test, y_pred), language="text")

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    st.divider()
    st.subheader("🔍 Test a Review")

    user_text = st.text_area("Enter review text:")

    if user_text:
        cleaned = clean_text(user_text)
        seq = tok.texts_to_sequences([cleaned])
        pad_seq = pad_sequences(seq, maxlen=MAX_LEN)
        prob = model.predict(pad_seq)[0][0]

        THRESHOLD = 0.40
        label = "AI-Generated" if prob >= THRESHOLD else "Original"
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
        # ------------------------
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

else:
    st.info("Train or load a model to enable evaluation and prediction.")
