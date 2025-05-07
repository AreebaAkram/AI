import streamlit as st
from textblob import TextBlob
st.title("Sentiment analysis")
text = st.text_area("enter your comment about the product")
blob = TextBlob(text)
# result = blob.sentiment
# if st.button("check"):
#     st.write(result)
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity
if st.button("Check polarity"):
    st.write(polarity)
    if (polarity > 0):
        st.write("Positive Comment")
    elif( polarity < 0 ):
        st.write("Negative Comment")
    else:
        st.write("Neutral Comment")
if st.button("check subjectivity"):
    st.write(subjectivity)
    if (subjectivity == 0):
        st.write("Completely objective")
    else:
        st.write("Highly subjective")