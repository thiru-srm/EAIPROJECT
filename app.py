import joblib
import streamlit as st

st.set_page_config(page_title="Spam Message Detector", page_icon="📩")

@st.cache_resource
def load_model():
    return joblib.load("spam_model.joblib")

model = load_model()

st.title("📩 AI Spam Message Detector")
st.write("Paste an SMS, WhatsApp or email message and the AI will tell you if it looks like spam.")

message = st.text_area("Your message", height=150, placeholder="e.g. You have won a free iPhone! Click here...")

if st.button("Check message"):
    if not message.strip():
        st.warning("Please type or paste a message first.")
    else:
        probs = model.predict_proba([message])[0]
        classes = list(model.classes_)
        spam_prob = probs[classes.index("spam")]

        if spam_prob >= 0.5:
            st.error(f"🚨 This looks like SPAM")
        else:
            st.success(f"✅ This looks SAFE (not spam)")

        st.metric("Spam probability", f"{spam_prob * 100:.1f}%")
        st.progress(float(spam_prob))
        st.caption("This is the model's estimate, not a guarantee. Never share OTPs or passwords, even if unsure.")
