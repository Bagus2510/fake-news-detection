import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
import altair as alt
from pathlib import Path
from streamlit_echarts import st_echarts

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
IMG_DIR = BASE_DIR / "images"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "model_best.pkl")
    tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    return model, tfidf


model, tfidf = load_model()


def remove_source_pattern(text):
    text = re.sub(r"\b[A-Z][A-Z ]+\s*\([A-Za-z]+\)\s*-\s*", "", text)
    text = re.sub(r"\([A-Za-z]+\)\s*-\s*", "", text)
    text = re.sub(r"\([A-Za-z]+ by[^)]*\)", "", text)
    text = re.sub(r"^[A-Z][A-Za-z ]+\s*-\s*", "", text)
    text = re.sub(r"\breuters\s*[-/?]?\s*ipsos\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthomson\s+reuters\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters\s*tv\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters\s+channel\w*\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\btold(?:\s+the)?\s+reuters\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\baccording to(?:\s+the)?\s+reuters\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bby reuters\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfrom reuters\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bat reuters\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bof reuters\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwith reuters\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bto reuters\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters has not\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters has been\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters has reported\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters reported\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters said\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters could not\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters was not\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters was unable\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters saw\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters spoke\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters sources\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters coverage\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters reporter\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters were\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters to\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters\s+report\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters\s+interview\w*\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters\s+poll\w*\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters\s+analysis\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters\s+data\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters editorial staff\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters news agency\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters journalists\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters reporters\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters foundation\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters blog\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters president\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\breuters\b", "", text, flags=re.IGNORECASE)
    return text


def preprocess(text):
    if not text or not isinstance(text, str):
        return ""
    text = remove_source_pattern(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"\S+\.com\S*", "", text)
    text = re.sub(r"\S+\.org\S*", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\ufffd", "", text)
    text = " ".join(text.split())
    return text


with st.sidebar:
    st.header("About")
    st.markdown(
        "Detect whether a news article is **Fake** or **Real** using "
        "a LinearSVC classifier trained on 44,689 articles."
    )
    st.divider()
    st.markdown("**Model**: LinearSVC (C=1.0)")
    st.markdown("**Accuracy**: 99.62%")
    st.markdown("**F1-Score**: 0.9962 (macro)")

    st.divider()
    st.markdown("### Model Performance")
    perf_data = {
        "Model": ["MultinomialNB", "LinearSVC", "RandomForest"],
        "Accuracy": [0.9573, 0.9962, 0.9875],
        "F1 (macro)": [0.9566, 0.9962, 0.9872],
    }
    st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)

    st.divider()
    st.caption("Built with Streamlit + Scikit-learn")


st.title("Fake News Detection")
st.markdown("Classify news articles as **Fake** or **Real** using machine learning.")

tab_predict, tab_eda, tab_about = st.tabs(["Predict", "EDA Insights", "About"])


def set_sample(title_val, text_val):
    st.session_state["title_input"] = title_val
    st.session_state["text_input"] = text_val


with tab_predict:
    st.subheader("Enter a news article")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Quick Test")
        st.markdown("Try a sample:")
        sample_news = {
            "Real - Factual News": (
                "The Federal Reserve held interest rates steady at 5.25%-5.50% "
                "on Wednesday, citing persistent inflation. Chair Jerome Powell "
                "said the central bank will not cut rates until data shows "
                "sustained progress toward its 2% target."
            ),
            "Real - Wire Report": (
                "Tokyo Japan - Toyota Motor Corp reported a 78% increase in "
                "quarterly operating profit on Thursday driven by strong demand "
                "for hybrid vehicles in North America and a weaker yen. The "
                "automaker maintained its annual profit forecast of 3 trillion yen."
            ),
            "Fake - Clickbait": (
                "SHOCKING: Scientists discover that the moon is actually "
                "hollow and government has been hiding the truth for decades! "
                "Whistleblower leaks classified documents proving what they "
                "dont want you to know!!"
            ),
            "Fake - Misleading": (
                "Breaking: Major bank announces ALL customer accounts will be "
                "frozen starting Monday due to new government regulation. "
                "Share this before they delete it! Your money is at risk and "
                "nobody is telling you about this!"
            ),
        }
        for label, sample in sample_news.items():
            title_val = label.split(" - ", 1)[1]
            st.button(
                label,
                use_container_width=True,
                on_click=set_sample,
                args=(title_val, sample),
            )

    with col1:
        title = st.text_input(
            "Title",
            key="title_input",
            placeholder="Enter the article title...",
        )
        text = st.text_area(
            "Article Body",
            key="text_input",
            height=250,
            placeholder="Paste the full article text here...",
        )

    st.divider()

    @st.fragment
    def predict_section():
        col_btn, col_result = st.columns([1, 2])

        with col_btn:
            predict_clicked = st.button(
                "Predict", type="primary", use_container_width=True
            )

        if predict_clicked:
            full_input = f"{title} {text}".strip()
            if not full_input:
                st.warning("Please enter a title or article body.")
                return

            status = st.status("Classifying...", expanded=False)
            clean = preprocess(full_input)
            vec = tfidf.transform([clean])
            pred = model.predict(vec)[0]
            label = "FAKE" if pred == 0 else "REAL"
            decision = model.decision_function(vec)[0]
            prob_fake = 1 / (1 + np.exp(decision))
            prob_real = 1 - prob_fake
            status.update(label=f"Prediction: {label}", state="complete", expanded=False)

            if label == "FAKE":
                st.error(f"### Prediction: {label}")
            else:
                st.success(f"### Prediction: {label}")

            st.subheader("Confidence Scores")

            gcol1, gcol2 = st.columns(2)

            with gcol1:
                gauge_option = {
                    "series": [
                        {
                            "type": "gauge",
                            "startAngle": 200,
                            "endAngle": -20,
                            "min": 0,
                            "max": 100,
                            "progress": {
                                "show": True,
                                "roundCap": True,
                                "width": 18,
                                "itemStyle": {
                                    "color": "#FF4B4B" if label == "FAKE" else "#28a745"
                                },
                            },
                            "axisLine": {
                                "lineStyle": {
                                    "width": 18,
                                    "color": [[1, "#e6e6e6"]],
                                }
                            },
                            "axisTick": {"show": False},
                            "splitLine": {"show": False},
                            "axisLabel": {"show": False},
                            "pointer": {"show": False},
                            "anchor": {"show": False},
                            "title": {"show": False},
                            "detail": {
                                "valueAnimation": True,
                                "formatter": "{value}%",
                                "fontSize": 32,
                                "fontWeight": "bold",
                                "color": "#FF4B4B" if label == "FAKE" else "#28a745",
                                "offsetCenter": [0, "0%"],
                            },
                            "data": [
                                {"value": round(prob_fake * 100 if label == "FAKE" else prob_real * 100, 1)}
                            ],
                        }
                    ],
                }
                st_echarts(gauge_option, height="250px")

            with gcol2:
                chart_data = pd.DataFrame(
                    {
                        "Class": ["Fake", "Real"],
                        "Score": [round(prob_fake * 100, 1), round(prob_real * 100, 1)],
                    }
                )
                bar = (
                    alt.Chart(chart_data)
                    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                    .encode(
                        x=alt.X("Class:N", axis=None),
                        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100])),
                        color=alt.Color(
                            "Class:N",
                            scale=alt.Scale(
                                domain=["Fake", "Real"],
                                range=["#FF4B4B", "#28a745"],
                            ),
                            legend=None,
                        ),
                    )
                    .properties(height=250)
                )
                text_chart = bar.mark_text(dy=-8, fontSize=14, fontWeight="bold").encode(
                    text="Score:Q"
                )
                st.altair_chart(bar + text_chart, use_container_width=True)

            st.subheader("Score Breakdown")
            radar_option = {
                "radar": {
                    "indicator": [
                        {"name": "Fake", "max": 100},
                        {"name": "Real", "max": 100},
                    ],
                    "shape": "circle",
                    "splitNumber": 5,
                },
                "series": [
                    {
                        "type": "radar",
                        "data": [
                            {
                                "value": [
                                    round(prob_fake * 100, 1),
                                    round(prob_real * 100, 1),
                                ],
                                "name": "Confidence",
                                "areaStyle": {
                                    "color": "rgba(255, 75, 75, 0.2)"
                                    if label == "FAKE"
                                    else "rgba(40, 167, 69, 0.2)"
                                },
                                "lineStyle": {
                                    "color": "#FF4B4B"
                                    if label == "FAKE"
                                    else "#28a745",
                                    "width": 2,
                                },
                                "itemStyle": {
                                    "color": "#FF4B4B"
                                    if label == "FAKE"
                                    else "#28a745"
                                },
                            }
                        ],
                    }
                ],
            }
            st_echarts(radar_option, height="300px")

            with st.expander("View preprocessed text"):
                st.code(clean, language="text")

    predict_section()


with tab_eda:
    st.subheader("Exploratory Data Analysis")

    st.markdown("### Label Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(IMG_DIR / "01_label_distribution.png"), use_container_width=True)
    with col2:
        st.image(str(IMG_DIR / "02_subject_leakage.png"), use_container_width=True)

    st.markdown("### Text Analysis")
    col3, col4 = st.columns(2)
    with col3:
        st.image(str(IMG_DIR / "03_text_length_distribution.png"), use_container_width=True)
    with col4:
        st.image(str(IMG_DIR / "04_top_words.png"), use_container_width=True)

    st.markdown("### Model Evaluation")
    col5, col6 = st.columns(2)
    with col5:
        st.image(str(IMG_DIR / "05_confusion_matrix_default.png"), use_container_width=True)
    with col6:
        st.image(str(IMG_DIR / "06_roc_curve_default.png"), use_container_width=True)

    col7, col8 = st.columns(2)
    with col7:
        st.image(str(IMG_DIR / "07_precision_recall_default.png"), use_container_width=True)
    with col8:
        st.image(str(IMG_DIR / "09_feature_importance_best.png"), use_container_width=True)


with tab_about:
    st.subheader("About")

    st.markdown(
        "Fake news classifier using LinearSVC on 44,689 news articles. "
        "TF-IDF vectorization with unigrams + bigrams, trained to 99.62% accuracy."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Dataset**")
        st.markdown(
            "- 23,478 Fake + 21,211 Real articles\n"
            "- Subject column dropped (data leakage)\n"
            "- Reuters source patterns removed via regex"
        )

    with col2:
        st.markdown("**Pipeline**")
        st.markdown(
            "- TF-IDF: ngram(1,2), 50k features, sublinear_tf\n"
            "- LinearSVC: C=1.0, max_iter=10000\n"
            "- Preprocessing: lowercase, URL/punctuation/digit removal"
        )

    st.markdown("---")
    st.markdown(
        "- [GitHub](https://github.com/Bagus2510/fake-news-detection) | "
        "- [LinkedIn](https://www.linkedin.com/in/bagusrahmadani/) | "
        "- [Dataset](https://www.kaggle.com/datasets/cbwarner/fake-or-real-news)"
    )
