import streamlit as st
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load tokenizer and model
tokenizer = pickle.load(open("tokenizer.pickle", "rb"))
model = load_model("cnn_model.keras")

st.markdown(
    """
    <style>
    .title {
        font-size: 42px !important;
    }
    .custom-label {
        font-size: 30px !important;
        font-weight: bold;
        margin-bottom: 0px !important;
    }
    .stTextArea textarea {
        font-size: 25px !important;
    }
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="title"> 📧 Email Spam Detector </p>', unsafe_allow_html=True)

st.write('<p class="custom-label">Enter email content:</p>', unsafe_allow_html=True)
email_text = st.text_area("",height=200)

if st.button("Check"):
    if email_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        sequence = tokenizer.texts_to_sequences([email_text])
        padded = pad_sequences(sequence, maxlen=100)
        prediction = model.predict(padded)[0][0]
        label = "🚫 Spam" if prediction > 0.5 else "✅ Not Spam"
        st.subheader(f"Prediction: {label}")