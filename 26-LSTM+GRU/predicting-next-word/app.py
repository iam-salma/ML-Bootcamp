import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('models/next_word_lstm.h5')

# Load the tokenizer
with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Unknown"

# ----------------------------
# Streamlit UI
st.set_page_config(page_title="Next Word Predictor", page_icon="🔮", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🔮 Next Word Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter a phrase, and let the LSTM model predict the next Shakespearean-style word.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input section
input_text = st.text_input("📘 Enter your sentence:", "To be or not to")

# Button in center
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("✨ Predict Next Word"):
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.success(f"🔤 **Next word:** `{next_word}`")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: lightgrey;'>Powered by LSTM · Built with 💙 Streamlit</p>", unsafe_allow_html=True)
