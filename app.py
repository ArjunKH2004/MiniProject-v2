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

# Initialize a session state variable to control the chat monitoring
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False

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

            # Initialize a list to store messages
            all_messages = []

            # Set monitoring to True
            st.session_state.monitoring = True

            while st.session_state.monitoring:
                chat_response = requests.get(chat_url, params=chat_params).json()
                if "items" in chat_response:
                    for item in chat_response["items"]:
                        author = item["authorDetails"]["displayName"]
                        message = item["snippet"]["displayMessage"]

                        # Classifying text
                        new_text = message
                        text_tfidf = tfidf.transform([new_text])
                        output = model.predict(text_tfidf)

                        # Coloring text based on classification
                        if output[0] == 0:
                            formatted_message = f"<span style='color:white;'>{author} : <span style='color:red;'>{message}</span></span>"
                        elif output[0] == 1:
                            formatted_message = f"<span style='color:white;'>{author} : <span style='color:yellow;'>{message}</span></span>"
                        elif output[0] == 2:
                            formatted_message = f"<span style='color:white;'>{author} : <span style='color:green;'>{message}</span></span>"

                        # Add the formatted message to the list
                        all_messages.append(formatted_message)

                    # Reverse the order of messages to show the newest on top
                    chat_placeholder.markdown("<br>".join(reversed(all_messages)), unsafe_allow_html=True)

                time.sleep(2)  # Wait before fetching new messages
        else:
            st.error("No live chat available for this video.")
    else:
        st.error("Please enter a valid YouTube video URL.")

# Button to pause the chat monitoring
if st.button("Pause Chat Monitoring"):
    st.session_state.monitoring = False
    st.write("Chat monitoring paused. Click 'Start Chat Monitoring' to resume.")
