import streamlit as st
import joblib

# Load the model
model = joblib.load('logistic_regression.pkl')

# Function to classify sentiment
def classify_sentiment(review):
    sentiment = model.predict([review])[0]
    return "Positive" if sentiment == 1 else "Negative"

# Get emoji based on sentiment
def get_sentiment_emoji(sentiment):
    return "😃" if sentiment == "Positive" else "😔"

# Streamlit UI
def main():
    st.set_page_config(page_title="Sentiment Analysis App", page_icon=":speech_balloon:")

    # Add custom CSS for background image
    st.markdown(
        """
        <style>
        body {
            background-image: url("https://editor.analyticsvidhya.com/uploads/13413surveysensum.png");
            background-size: cover;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Sentiment Analysis App")
    
    # Text input for entering review
    review = st.text_area("Enter your review below:", height=200)
    
    # Button to trigger sentiment analysis
    if st.button("Submit", key="submit_button"):
        if review.strip() == "":
            st.error("Please enter a review.")
        else:
            with st.spinner("Analyzing..."):
                result = classify_sentiment(review)
                sentiment_emoji = get_sentiment_emoji(result)
                st.success(f"Sentiment: {result} {sentiment_emoji}")
                st.markdown("---")
                st.write("### Results")
                st.write("Here are the details of the analysis:")
                st.write(f"- **Sentiment:** {result}")
                st.write(f"- **Emoji:** {sentiment_emoji}")

if __name__ == "__main__":
    main()
