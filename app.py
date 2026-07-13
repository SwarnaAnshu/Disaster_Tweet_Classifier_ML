import gradio as gr
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

model = joblib.load("trained_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


def predict_disaster(tweet):
    # No input entered -> show nothing instead of a meaningless prediction
    if not tweet or not tweet.strip():
        return (
            gr.update(value="Please enter a tweet to classify.", visible=True),
            gr.update(value="", visible=False),
        )

    cleaned = clean_text(tweet)

    if not cleaned:
        return (
            gr.update(value="Tweet contains no usable text after cleaning — please enter a valid tweet.", visible=True),
            gr.update(value="", visible=False),
        )

    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][1]

    if pred == 1:
        label_html = f"""
        <div style="background-color:#3a1f1f; border:1px solid #ff4d4d; border-radius:8px;
                    padding:14px 18px; font-size:18px; font-weight:600; color:#ff6b6b;">
            🚨 Real Disaster
        </div>
        """
        confidence = proba
        confidence_label = "Real Disaster Confidence"
        bar_color = "#ff4d4d"
    else:
        label_html = f"""
        <div style="background-color:#1f3a24; border:1px solid #4caf50; border-radius:8px;
                    padding:14px 18px; font-size:18px; font-weight:600; color:#66d17a;">
            ✅ Not a Disaster
        </div>
        """
        confidence = 1 - proba
        confidence_label = "Not a Disaster Confidence"
        bar_color = "#4caf50"

    pct = round(confidence * 100, 2)

    confidence_html = f"""
    <div style="margin-top:10px;">
        <div style="font-size:14px; color:#aaaaaa; margin-bottom:6px;">{confidence_label}</div>
        <div style="font-size:30px; font-weight:700; margin-bottom:8px;">{pct}%</div>
        <div style="background-color:#333333; border-radius:6px; height:12px; width:100%; overflow:hidden;">
            <div style="background-color:{bar_color}; height:100%; width:{pct}%;"></div>
        </div>
    </div>
    """

    return (
        gr.update(value=label_html, visible=True),
        gr.update(value=confidence_html, visible=True),
    )


custom_css = """
#label-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
}
#clear-btn {
    max-width: 100px !important;
    width: 100px !important;
    flex-grow: 0 !important;
    flex-basis: 100px !important;
}
#predict-row {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
}
#predict-btn {
    max-width: 160px !important;
    width: 160px !important;
    flex-grow: 0 !important;
    flex-basis: 160px !important;
    margin: 0 auto !important;
}
footer {
    display: none !important;
}
"""

with gr.Blocks(title="DisasterAlert: Disaster Tweet Classifier") as demo:
    with gr.Column():
        gr.Markdown("# 🌪️ DisasterAlert: Disaster Tweet Classifier")
        gr.Markdown("Predict whether a tweet refers to a **real disaster** or **not**, using Machine Learning Model.")

        with gr.Row(elem_id="label-row"):
            gr.Markdown("### Enter Tweet Text")
            clear_btn = gr.Button("Clear", size="sm", elem_id="clear-btn")

        tweet_input = gr.Textbox(
            lines=6,
            placeholder="Type or paste your tweet here...",
            show_label=False,
        )

        with gr.Row(elem_id="predict-row"):
            predict_btn = gr.Button("Predict", variant="primary", size="sm", elem_id="predict-btn")

        result_box = gr.HTML(visible=False)
        confidence_bar = gr.HTML(visible=False)

        gr.Markdown("---")
        gr.Markdown("Built with Scikit-Learn | TF-IDF | Multinomial Naive Bayes")

        predict_btn.click(
            fn=predict_disaster,
            inputs=tweet_input,
            outputs=[result_box, confidence_bar],
        )

        clear_btn.click(
            fn=lambda: ("", gr.update(value="", visible=False), gr.update(value="", visible=False)),
            inputs=None,
            outputs=[tweet_input, result_box, confidence_bar],
        )

demo.launch(css=custom_css, footer_links=[])