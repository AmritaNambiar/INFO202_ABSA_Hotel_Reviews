import streamlit as st
import pandas as pd
from absa_model import predict_aspect_sentiments

st.set_page_config(
    page_title="Hotel Aspect-Based Sentiment Explorer",
    page_icon="🏨",
    layout="wide",
)

st.title("🏨 Hotel Aspect-Based Sentiment Explorer")
st.caption(
    "Drop a hotel review or upload a CSV and see aspect-level sentiment for "
    "Location, Room, Cleanliness, Service, Facilities, Food & Beverage, Price, and Safety."
)

tab_single, tab_batch = st.tabs(["Single review", "Batch CSV analysis"])

# --- Single review ---
with tab_single:
    st.subheader("Single review analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        default_text = (
            "Drop a hotel review here!"
        )
        review = st.text_area(
            "Hotel review text",
            value=default_text,
            height=220,
        )

    if st.button("Analyze review", type="primary"):
        if not review.strip():
            st.warning("Please paste a review first.")
        else:
            with st.spinner("Running aspect-based sentiment analysis..."):
                results = predict_aspect_sentiments(review)

            # Show results
            st.subheader("Aspect breakdown")
            rows = []
            for aspect, (label, conf) in results.items():
                rows.append(
                    {
                        "Aspect": aspect,
                        "Sentiment": label,
                        "Confidence": round(conf, 2),
                    }
                )
            df_out = pd.DataFrame(rows)
            st.table(df_out)

# --- Batch CSV mode ---
with tab_batch:
    st.subheader("Batch CSV analysis")
    st.write("Upload a CSV with a `review_text` column to analyze multiple reviews.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "review_text" not in df.columns:
            st.error("CSV must contain a `review_text` column.")
        else:
            st.success(f"Loaded {len(df)} reviews.")
            max_n = st.number_input(
                "Max reviews to analyze",
                min_value=1,
                max_value=len(df),
                value=min(50, len(df)),
            )

            if st.button("Run batch ABSA", type="primary"):
                subset = df.head(int(max_n)).copy()

                batch_rows = []
                with st.spinner("Analyzing reviews..."):
                    for i, row in subset.iterrows():
                        text = str(row["review_text"])
                        res = predict_aspect_sentiments(text)
                        flat = {"row_index": i, "review_text": text}
                        for aspect, (label, conf) in res.items():
                            flat[f"{aspect}_sentiment"] = label
                            flat[f"{aspect}_conf"] = round(conf, 2)
                        batch_rows.append(flat)

                df_res = pd.DataFrame(batch_rows)
                st.dataframe(df_res, use_container_width=True)

                csv_bytes = df_res.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results CSV",
                    data=csv_bytes,
                    file_name="absa_hotel_results.csv",
                    mime="text/csv",
                )
