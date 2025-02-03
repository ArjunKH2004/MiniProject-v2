import requests
import time
import pickle
import streamlit as st

# Load the model and vectorizer
with open("yt_ai_classifier_model_2.sav", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.sav", "rb") as f:
    tfidf = pickle.load(f)

# Streamlit app
st.title("YouTube Live Chat Classifier")

API_KEY = "AIzaSyAiggkZTjKRGd_3pYRTJIe-08QOjfnHYgI"
video_url_input = st.text_input("Enter the YouTube video URL:")

if st.button("Start Chat Monitoring"):
    if video_url_input:
        # Extract VIDEO_ID from the URL
        VIDEO_ID = video_url_input.split('v=')[1].split('&')[0] if 'v=' in video_url_input else video_url_input.split('/')[-1]

        video_url = "https://www.googleapis.com/youtube/v3/videos"
        video_params = {
            "part": "liveStreamingDetails",
            "id": VIDEO_ID,
            "key": API_KEY
        }
        response = requests.get(video_url, params=video_params).json()

        if "items" in response and response["items"]:
            live_chat_id = response["items"][0]["liveStreamingDetails"]["activeLiveChatId"]
            chat_url = "https://www.googleapis.com/youtube/v3/liveChat/messages"
            chat_params = {
                "liveChatId": live_chat_id,
                "part": "snippet,authorDetails",
                "key": API_KEY
            }

            # Start monitoring chat
            st.write("Monitoring live chat...")

            # Create a placeholder for chat messages
            chat_placeholder = st.empty()

            while True:
                chat_response = requests.get(chat_url, params=chat_params).json()
                if "items" in chat_response:
                    messages = []
                    for item in chat_response["items"]:
                        author = item["authorDetails"]["displayName"]
                        message = item["snippet"]["displayMessage"]

                        # Classifying text
                        new_text = message
                        text_tfidf = tfidf.transform([new_text])
                        output = model.predict(text_tfidf)

                        # Coloring text based on classification
                        if output[0] == 0:
                            messages.append(f"<span style='color:red;'>{author} : {message}</span>")
                        elif output[0] == 1:
                            messages.append(f"<span style='color:yellow;'>{author} : {message}</span>")
                        elif output[0] == 2:
                            messages.append(f"<span style='color:green;'>{author} : {message}</span>")

                    # Update the chat placeholder with new messages
                    chat_placeholder.markdown("<br>".join(messages), unsafe_allow_html=True)

                time.sleep(2)  # Wait before fetching new messages
        else:
            st.error("No live chat available for this video.")
    else:
        st.error("Please enter a valid YouTube video URL.")
