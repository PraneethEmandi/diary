# # # import streamlit as st
# # # from txtai.pipeline import Summary, Textractor
# # # from PyPDF2 import PdfReader
# # # import joblib
# # # import pandas as pd
# # # import numpy as np
# # # import altair as alt
# # #
# # # st.set_page_config(layout="wide")
# # # pipe_lr = joblib.load(open("model/text_emotions.pkl", "rb"))
# # #
# # # @st.cache_resource
# # # def text_summary(text, maxlength=None):
# # #     summary = Summary()
# # #     result = summary(text)
# # #     return result
# # #
# # # def extract_text_from_pdf(file_path):
# # #     with open(file_path, "rb") as f:
# # #         reader = PdfReader(f)
# # #         page = reader.pages[0]
# # #         text = page.extract_text()
# # #     return text
# # #
# # # def predict_emotions(docx):
# # #     results = pipe_lr.predict([docx])
# # #     return results[0]
# # #
# # # def get_prediction_proba(docx):
# # #     results = pipe_lr.predict_proba([docx])
# # #     return results
# # #
# # # emotions_emoji_dict = {
# # #     "anger": "😠", "disgust": "🤮", "fear": "😨😱",
# # #     "happy": "🤗", "joy": "😂", "neutral": "😐",
# # #     "sad": "😔", "sadness": "😔", "shame": "😳",
# # #     "surprise": "😮"
# # # }
# # #
# # # def main():
# # #     st.title("Text Summarization and Emotion Detection")
# # #
# # #     choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])
# # #
# # #     if choice == "Summarize Text":
# # #         st.subheader("Summarize Text using txtai")
# # #         input_text = st.text_area("Enter your text here")
# # #         if st.button("Summarize Text"):
# # #             col1, col2, col3 = st.columns([1, 1, 1])
# # #             with col1:
# # #                 st.markdown("**Your Input Text**")
# # #                 st.info(input_text)
# # #             with col2:
# # #                 st.markdown("**Summary Result**")
# # #                 result = text_summary(input_text)
# # #                 st.success(result)
# # #             with col3:
# # #                 st.markdown("**Emotion Prediction**")
# # #                 prediction = predict_emotions(input_text)
# # #                 probability = get_prediction_proba(input_text)
# # #                 emoji_icon = emotions_emoji_dict[prediction]
# # #                 st.write("{}:{}".format(prediction, emoji_icon))
# # #                 st.write("Confidence:{}".format(np.max(probability)))
# # #
# # #                 st.success("Prediction Probability")
# # #                 proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
# # #                 proba_df_clean = proba_df.T.reset_index()
# # #                 proba_df_clean.columns = ["emotions", "probability"]
# # #                 fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
# # #                 st.altair_chart(fig, use_container_width=True)
# # #
# #     # elif choice == "Summarize Document":
# #     #     st.subheader("Summarize Document using txtai")
# #     #     input_file = st.file_uploader("Upload your document here", type=['pdf'])
# #     #     if st.button("Summarize Document"):
# #     #         with open("doc_file.pdf", "wb") as f:
# #     #             f.write(input_file.getbuffer())
# #     #         col1, col2, col3 = st.columns([1, 1, 1])
# #     #         with col1:
# #     #             st.info("File uploaded successfully")
# #     #             extracted_text = extract_text_from_pdf("doc_file.pdf")
# #     #             st.markdown("**Extracted Text is Below:**")
# #     #             st.info(extracted_text)
# #     #         with col2:
# #     #             st.markdown("**Summary Result**")
# #     #             doc_summary = text_summary(extracted_text)
# #     #             st.success(doc_summary)
# #     #         with col3:
# #     #             st.markdown("**Emotion Prediction**")
# #     #             prediction = predict_emotions(extracted_text)
# #     #             probability = get_prediction_proba(extracted_text)
# #     #             emoji_icon = emotions_emoji_dict[prediction]
# #     #             st.write("{}:{}".format(prediction, emoji_icon))
# #     #             st.write("Confidence:{}".format(np.max(probability)))
# #     #
# #     #             st.success("Prediction Probability")
# #     #             proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
# #     #             proba_df_clean = proba_df.T.reset_index()
# #     #             proba_df_clean.columns = ["emotions", "probability"]
# #     #             fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
# #     #             st.altair_chart(fig, use_container_width=True)
# # #
# # # if __name__ == '__main__':
# # #     main()
# # import os
# # import streamlit as st
# # import assemblyai as aai
# # from txtai.pipeline import Summary
# # import joblib
# # import pandas as pd
# # import numpy as np
# # import altair as alt
# #
# # st.set_page_config(layout="wide")
# # pipe_lr = joblib.load(open("model/text_emotions.pkl", "rb"))
# #
# # @st.cache_resource
# # def text_summary(text, maxlength=None):
# #     summary = Summary()
# #     result = summary(text)
# #     return result
# #
# # def predict_emotions(docx):
# #     results = pipe_lr.predict([docx])
# #     return results[0]
# #
# # def get_prediction_proba(docx):
# #     results = pipe_lr.predict_proba([docx])
# #     return results
# #
# # emotions_emoji_dict = {
# #     "anger": "😠", "disgust": "🤮", "fear": "😨😱",
# #     "happy": "🤗", "joy": "😂", "neutral": "😐",
# #     "sad": "😔", "sadness": "😔", "shame": "😳",
# #     "surprise": "😮"
# # }
# #
# # def main():
# #     st.title("Text Summarization and Emotion Detection")
# #
# #     choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document", "Summarize Audio"])
# #
# #     if choice == "Summarize Text":
# #         st.subheader("Summarize Text using txtai")
# #         input_text = st.text_area("Enter your text here")
# #         if st.button("Summarize Text"):
# #             col1, col2, col3 = st.columns([1, 1, 1])
# #             with col1:
# #                 st.markdown("**Your Input Text**")
# #                 st.info(input_text)
# #             with col2:
# #                 st.markdown("**Summary Result**")
# #                 result = text_summary(input_text)
# #                 st.success(result)
# #             with col3:
# #                 st.markdown("**Emotion Prediction**")
# #                 prediction = predict_emotions(input_text)
# #                 probability = get_prediction_proba(input_text)
# #                 emoji_icon = emotions_emoji_dict[prediction]
# #                 st.write("{}:{}".format(prediction, emoji_icon))
# #                 st.write("Confidence:{}".format(np.max(probability)))
# #
# #                 st.success("Prediction Probability")
# #                 proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
# #                 proba_df_clean = proba_df.T.reset_index()
# #                 proba_df_clean.columns = ["emotions", "probability"]
# #                 fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
# #                 st.altair_chart(fig, use_container_width=True)
# #
# #
# #     elif choice == "Summarize Document":
# #
# #         st.subheader("Summarize Document using txtai")
# #
# #         input_file = st.file_uploader("Upload your document here", type=['pdf'])
# #
# #         if st.button("Summarize Document"):
# #             with open("doc_file.pdf", "wb") as f:
# #                 f.write(input_file.getbuffer())
# #
# #             col1, col2, col3 = st.columns([1, 1, 1])
# #
# #             with col1:
# #                 st.info("File uploaded successfully")
# #
# #                 extracted_text = extract_text_from_pdf("doc_file.pdf")
# #
# #                 st.markdown("**Extracted Text is Below:**")
# #
# #                 st.info(extracted_text)
# #
# #             with col2:
# #                 st.markdown("**Summary Result**")
# #
# #                 doc_summary = text_summary(extracted_text)
# #
# #                 st.success(doc_summary)
# #
# #             with col3:
# #                 st.markdown("**Emotion Prediction**")
# #
# #                 prediction = predict_emotions(extracted_text)
# #
# #                 probability = get_prediction_proba(extracted_text)
# #
# #                 emoji_icon = emotions_emoji_dict[prediction]
# #
# #                 st.write("{}:{}".format(prediction, emoji_icon))
# #
# #                 st.write("Confidence:{}".format(np.max(probability)))
# #
# #                 st.success("Prediction Probability")
# #
# #                 proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
# #
# #                 proba_df_clean = proba_df.T.reset_index()
# #
# #                 proba_df_clean.columns = ["emotions", "probability"]
# #
# #                 fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
# #
# #                 st.altair_chart(fig, use_container_width=True)
# #
# #
# #     elif choice == "Summarize Audio":
# #
# #         st.subheader("Summarize Audio using AssemblyAI")
# #
# #         uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
# #
# #         if uploaded_file:
# #             audio_bytes = uploaded_file.read()
# #
# #             audio_path = "audio_file.wav"  # Temporary file path for audio file
# #
# #             with open(audio_path, "wb") as f:
# #                 f.write(audio_bytes)
# #
# #             st.audio(audio_path, format="audio/wav")
# #
# #             # Transcribe the audio file
# #
# #             aai.settings.api_key = "ed9174d8ec5a45afa0075b544b8eb2d7"
# #
# #             transcriber = aai.Transcriber()
# #
# #             transcript = transcriber.transcribe(audio_path)
# #
# #             # Display the transcription summary
# #
# #             st.subheader("Transcription Summary")
# #
# #             transcript_summary = text_summary(transcript.text)
# #
# #             st.success(transcript_summary)
# #
# #             # Display the emotion prediction
# #
# #             st.subheader("Emotion Prediction")
# #
# #             prediction = predict_emotions(transcript.text)
# #
# #             probability = get_prediction_proba(transcript.text)
# #
# #             emoji_icon = emotions_emoji_dict[prediction]
# #
# #             st.write("{}:{}".format(prediction, emoji_icon))
# #
# #             st.write("Confidence:{}".format(np.max(probability)))
# #
# #             # Display the prediction probability graph
# #
# #             st.success("Prediction Probability")
# #
# #             proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
# #
# #             proba_df_clean = proba_df.T.reset_index()
# #
# #             proba_df_clean.columns = ["emotions", "probability"]
# #
# #             fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
# #
# #             st.altair_chart(fig, use_container_width=True)
# #
# # if __name__ == '__main__':
# #     main()
# import pyrebase
# from datetime import datetime
# import os
# import streamlit as st
# import assemblyai as aai
# from txtai.pipeline import Summary
# import joblib
# import pandas as pd
# import numpy as np
# import altair as alt
#
# # import other necessary libraries and functions
#
# # Configuration Key
# import pyrebase
# import streamlit as st
# from datetime import datetime
# import joblib
# import pandas as pd
# import numpy as np
# import altair as alt
#
# st.set_page_config(layout="wide")
#
# # Configuration Key
# firebaseConfig = {
#   'apiKey': "AIzaSyDBM_FNVgKZPKINBwFEryTu-Nb6FLD3UH4",
#   'authDomain': "godsake-c3bd9.firebaseapp.com",
#   'projectId': "godsake-c3bd9",
#   'databaseURL' : "https://godsake-c3bd9-default-rtdb.europe-west1.firebasedatabase.app/",
#   'storageBucket': "godsake-c3bd9.appspot.com",
#   'messagingSenderId': "388373732392",
#   'appId': "1:388373732392:web:16fb742f3c5adaa2d8a91d",
#   'measurementId': "G-WEYN9F6Q3T"
# }
#
# # Firebase Authentication
# firebase = pyrebase.initialize_app(firebaseConfig)
# auth = firebase.auth()
#
# # Database
# db = firebase.database()
# storage = firebase.storage()
# st.sidebar.title("Our community app")
#
# # Authentication
# choice = st.sidebar.selectbox('login/Signup', ['Login', 'Sign up'])
#
# # Obtain User Input for email and password
# email = st.sidebar.text_input('Please enter your email address')
# password = st.sidebar.text_input('Please enter your password', type='password')
#
# # App
# pipe_lr = joblib.load(open("model/text_emotions.pkl", "rb"))
#
# # Continue with the rest of your code...
#
#
# @st.cache_resource
# def text_summary(text, maxlength=None):
#     summary = Summary()
#     result = summary(text)
#     return result
#
# def predict_emotions(docx):
#     results = pipe_lr.predict([docx])
#     return results[0]
#
# def get_prediction_proba(docx):
#     results = pipe_lr.predict_proba([docx])
#     return results
#
# emotions_emoji_dict = {
#     "anger": "😠", "disgust": "🤮", "fear": "😨😱",
#     "happy": "🤗", "joy": "😂", "neutral": "😐",
#     "sad": "😔", "sadness": "😔", "shame": "😳",
#     "surprise": "😮"
# }
#
#
# # Sign up Block
# if choice == 'Sign up':
#     handle = st.sidebar.text_input(
#         'Please input your app handle name', value='Default')
#     submit = st.sidebar.button('Create my account')
#
#     if submit:
#         user = auth.create_user_with_email_and_password(email, password)
#         st.success('Your account is created suceesfully!')
#         st.balloons()
#         # Sign in
#         user = auth.sign_in_with_email_and_password(email, password)
#         db.child(user['localId']).child("Handle").set(handle)
#         db.child(user['localId']).child("ID").set(user['localId'])
#         st.title('Welcome' + handle)
#         st.info('Login via login drop down selection')
#
# # Login Block
# if choice == 'Login':
#     login = st.sidebar.checkbox('Login')
#     if login:
#         user = auth.sign_in_with_email_and_password(email, password)
#         st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
#         bio = st.radio('Jump to', ['Home', 'Settings', 'Summarize Text', 'Summarize Document', 'Summarize Audio'])
#
#         # SETTINGS PAGE
#         if bio == 'Settings':
#             # CHECK FOR IMAGE
#             nImage = db.child(user['localId']).child("Image").get().val()
#             # IMAGE FOUND
#             if nImage is not None:
#                 # We plan to store all our image under the child image
#                 Image = db.child(user['localId']).child("Image").get()
#                 for img in Image.each():
#                     img_choice = img.val()
#                     # st.write(img_choice)
#                 st.image(img_choice)
#                 exp = st.expander('Change Bio and Image')
#                 # User plan to change profile picture
#                 with exp:
#                     newImgPath = st.text_input('Enter full path of your profile imgae')
#                     upload_new = st.button('Upload')
#                     if upload_new:
#                         uid = user['localId']
#                         fireb_upload = storage.child(uid).put(newImgPath, user['idToken'])
#                         a_imgdata_url = storage.child(uid).get_url(fireb_upload['downloadTokens'])
#                         db.child(user['localId']).child("Image").push(a_imgdata_url)
#                         st.success('Success!')
#                         # IF THERE IS NO IMAGE
#             else:
#                 st.info("No profile picture yet")
#                 newImgPath = st.text_input('Enter full path of your profile image')
#                 upload_new = st.button('Upload')
#                 if upload_new:
#                     uid = user['localId']
#                     # Stored Initated Bucket in Firebase
#                     fireb_upload = storage.child(uid).put(newImgPath, user['idToken'])
#                     # Get the url for easy access
#                     a_imgdata_url = storage.child(uid).get_url(fireb_upload['downloadTokens'])
#                     # Put it in our real time database
#                     db.child(user['localId']).child("Image").push(a_imgdata_url)
#
#
#
#         # HOME PAGE
#         elif bio == 'Home':
#             # Your code for the home page here
#             col1, col2 = st.columns(2)
#
#             # col for Profile picture
#             with col1:
#                 nImage = db.child(user['localId']).child("Image").get().val()
#                 if nImage is not None:
#                     val = db.child(user['localId']).child("Image").get()
#                     for img in val.each():
#                         img_choice = img.val()
#                     st.image(img_choice, use_column_width=True)
#                 else:
#                     st.info("No profile picture yet. Go to Edit Profile and choose one!")
#
#                 post = st.text_input("Let's share my current mood as a post!", max_chars=100)
#                 add_post = st.button('Share Posts')
#             if add_post:
#                 now = datetime.now()
#                 dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
#                 post = {'Post:': post,
#                         'Timestamp': dt_string}
#                 results = db.child(user['localId']).child("Posts").push(post)
#                 st.balloons()
#
#             # This coloumn for the post Display
#             with col2:
#
#                 all_posts = db.child(user['localId']).child("Posts").get()
#                 if all_posts.val() is not None:
#                     for Posts in reversed(all_posts.each()):
#                         # st.write(Posts.key()) # Morty
#                         st.code(Posts.val(), language='')
#         # SUMMARIZE TEXT
#         elif bio == 'Summarize Text':
#             # Your code for summarizing text here
#             st.subheader("Summarize Text using txtai")
#             input_text = st.text_area("Enter your text here")
#             if st.button("Summarize Text"):
#                 col1, col2, col3 = st.columns([1, 1, 1])
#                 with col1:
#                     st.markdown("**Your Input Text**")
#                     st.info(input_text)
#                 with col2:
#                     st.markdown("**Summary Result**")
#                     result = text_summary(input_text)
#                     st.success(result)
#                 with col3:
#                     st.markdown("**Emotion Prediction**")
#                     prediction = predict_emotions(input_text)
#                     probability = get_prediction_proba(input_text)
#                     emoji_icon = emotions_emoji_dict[prediction]
#                     st.write("{}:{}".format(prediction, emoji_icon))
#                     st.write("Confidence:{}".format(np.max(probability)))
#
#                     st.success("Prediction Probability")
#                     proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
#                     proba_df_clean = proba_df.T.reset_index()
#                     proba_df_clean.columns = ["emotions", "probability"]
#                     fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
#                     st.altair_chart(fig, use_container_width=True)
#         # SUMMARIZE DOCUMENT
#         elif bio == 'Summarize Document':
#             # Your code for summarizing document here
#             st.subheader("Summarize Document using txtai")
#
#             input_file = st.file_uploader("Upload your document here", type=['pdf'])
#
#             if st.button("Summarize Document"):
#                 with open("doc_file.pdf", "wb") as f:
#                     f.write(input_file.getbuffer())
#
#                 col1, col2, col3 = st.columns([1, 1, 1])
#
#                 with col1:
#                     st.info("File uploaded successfully")
#
#                     extracted_text = extract_text_from_pdf("doc_file.pdf")
#
#                     st.markdown("**Extracted Text is Below:**")
#
#                     st.info(extracted_text)
#
#                 with col2:
#                     st.markdown("**Summary Result**")
#
#                     doc_summary = text_summary(extracted_text)
#
#                     st.success(doc_summary)
#
#                 with col3:
#                     st.markdown("**Emotion Prediction**")
#
#                     prediction = predict_emotions(extracted_text)
#
#                     probability = get_prediction_proba(extracted_text)
#
#                     emoji_icon = emotions_emoji_dict[prediction]
#
#                     st.write("{}:{}".format(prediction, emoji_icon))
#
#                     st.write("Confidence:{}".format(np.max(probability)))
#
#                     st.success("Prediction Probability")
#
#                     proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
#
#                     proba_df_clean = proba_df.T.reset_index()
#
#                     proba_df_clean.columns = ["emotions", "probability"]
#
#                     fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
#
#                     st.altair_chart(fig, use_container_width=True)
#             # SUMMARIZE AUDIO
#         elif bio == 'Summarize Audio':
#             # Your code for summarizing audio here
#             st.subheader("Summarize Audio using AssemblyAI")
#
#             uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
#
#             if uploaded_file:
#                 audio_bytes = uploaded_file.read()
#
#                 audio_path = "audio_file.wav"  # Temporary file path for audio file
#
#                 with open(audio_path, "wb") as f:
#                     f.write(audio_bytes)
#
#                 st.audio(audio_path, format="audio/wav")
#
#                 # Transcribe the audio file
#
#                 aai.settings.api_key = "ed9174d8ec5a45afa0075b544b8eb2d7"
#
#                 transcriber = aai.Transcriber()
#
#                 transcript = transcriber.transcribe(audio_path)
#
#                 # Display the transcription summary
#
#                 st.subheader("Transcription Summary")
#
#                 transcript_summary = text_summary(transcript.text)
#
#                 st.success(transcript_summary)
#
#                 # Display the emotion prediction
#
#                 st.subheader("Emotion Prediction")
#
#                 prediction = predict_emotions(transcript.text)
#
#                 probability = get_prediction_proba(transcript.text)
#
#                 emoji_icon = emotions_emoji_dict[prediction]
#
#                 st.write("{}:{}".format(prediction, emoji_icon))
#
#                 st.write("Confidence:{}".format(np.max(probability)))
#
#                 # Display the prediction probability graph
#
#                 st.success("Prediction Probability")
#
#                 proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
#
#                 proba_df_clean = proba_df.T.reset_index()
#
#                 proba_df_clean.columns = ["emotions", "probability"]
#
#                 fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
#
#                 st.altair_chart(fig, use_container_width=True)

# # if __name__ == '__main__':
# #     main()

import pyrebase
import streamlit as st
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import altair as alt
import os
import assemblyai as aai
from txtai.pipeline import Summary
from PyPDF2 import PdfReader
import sounddevice as sd  # Add this import statement
import wavio
import base64
st.set_page_config(layout="wide")




# Configuration Key
firebaseConfig = {
  'apiKey': "AIzaSyDBM_FNVgKZPKINBwFEryTu-Nb6FLD3UH4",
  'authDomain': "godsake-c3bd9.firebaseapp.com",
  'projectId': "godsake-c3bd9",
  'databaseURL' : "https://godsake-c3bd9-default-rtdb.europe-west1.firebasedatabase.app/",
  'storageBucket': "godsake-c3bd9.appspot.com",
  'messagingSenderId': "388373732392",
  'appId': "1:388373732392:web:16fb742f3c5adaa2d8a91d",
  'measurementId': "G-WEYN9F6Q3T"
}

# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()
st.sidebar.title("Our community app")

# Authentication
choice = st.sidebar.selectbox('login/Signup', ['Login', 'Sign up'])

# Obtain User Input for email and password
email = st.sidebar.text_input('Please enter your email address')
password = st.sidebar.text_input('Please enter your password', type='password')

# App
pipe_lr = joblib.load(open("model/text_emotions.pkl", "rb"))
def record_audio(seconds, fs, channels):
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=channels, dtype='int16')
    sd.wait()
    return recording
@st.cache_resource
def text_summary(text, maxlength=None):
    summary = Summary()
    result = summary(text)
    # Store result in database
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    db.child(user['localId']).child("Summaries").push({
        'Text': text,
        'Summary': result,
        'Timestamp': dt_string
    })
    return result
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    # Store result in database
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    db.child(user['localId']).child("Emotions").push({
        'Text': docx,
        'Prediction': results[0],
        'Timestamp': dt_string
    })
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    # Convert NumPy array to Python list
    results_list = results[0].tolist()
    # Store result in database
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    db.child(user['localId']).child("Emotions").push({
        'Text': docx,
        'Prediction': results_list,
        'Timestamp': dt_string
    })
    # Reshape the data for DataFrame
    results_reshaped = np.reshape(results_list, (-1, len(pipe_lr.classes_)))
    return results_reshaped

# Display the prediction probability graph


emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱",
    "happy": "🤗", "joy": "😂", "neutral": "😐",
    "sad": "😔", "sadness": "😔", "shame": "😳",
    "surprise": "😮"
}
# Sign up Block
if choice == 'Sign up':
    handle = st.sidebar.text_input(
        'Please input your app handle name', value='Default')
    submit = st.sidebar.button('Create my account')

    if submit:
        user = auth.create_user_with_email_and_password(email, password)
        st.success('Your account is created suceesfully!')
        st.balloons()
        # Sign in
        user = auth.sign_in_with_email_and_password(email, password)
        db.child(user['localId']).child("Handle").set(handle)
        db.child(user['localId']).child("ID").set(user['localId'])
        st.title('Welcome' + handle)
        st.info('Login via login drop down selection')

# Login Block
if choice == 'Login':
    login = st.sidebar.checkbox('Login')
    if login:
        user = auth.sign_in_with_email_and_password(email, password)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        bio = st.radio('Jump to', ['Home', 'Settings', 'Summarize Text', 'Summarize Document', 'Summarize Audio'])

        # SETTINGS PAGE
        if bio == 'Settings':
            # CHECK FOR IMAGE
            nImage = db.child(user['localId']).child("Image").get().val()
            # IMAGE FOUND
            if nImage is not None:
                # We plan to store all our image under the child image
                Image = db.child(user['localId']).child("Image").get()
                for img in Image.each():
                    img_choice = img.val()
                    # st.write(img_choice)
                st.image(img_choice)
                exp = st.expander('Change Bio and Image')
                # User plan to change profile picture
                with exp:
                    newImgPath = st.text_input('Enter full path of your profile imgae')
                    upload_new = st.button('Upload')
                    if upload_new:
                        uid = user['localId']
                        fireb_upload = storage.child(uid).put(newImgPath, user['idToken'])
                        a_imgdata_url = storage.child(uid).get_url(fireb_upload['downloadTokens'])
                        db.child(user['localId']).child("Image").push(a_imgdata_url)
                        st.success('Success!')
                        # IF THERE IS NO IMAGE
            else:
                st.info("No profile picture yet")
                newImgPath = st.text_input('Enter full path of your profile image')
                upload_new = st.button('Upload')
                if upload_new:
                    uid = user['localId']
                    # Stored Initated Bucket in Firebase
                    fireb_upload = storage.child(uid).put(newImgPath, user['idToken'])
                    # Get the url for easy access
                    a_imgdata_url = storage.child(uid).get_url(fireb_upload['downloadTokens'])
                    # Put it in our real time database
                    db.child(user['localId']).child("Image").push(a_imgdata_url)

# Display stored summaries and predictions on the home page
        # Display stored summaries and predictions on the home page
        # Display stored summaries and predictions on the home page




        elif bio == 'Home':
            # Your code for the home page here
            col1, col2 = st.columns(2)

            # Display profile picture
            with col1:
                nImage = db.child(user['localId']).child("Image").get().val()
                if nImage is not None:
                    val = db.child(user['localId']).child("Image").get()
                    for img in val.each():
                        img_choice = img.val()
                    st.image(img_choice, use_column_width=True)
                else:
                    st.info("No profile picture yet. Go to Edit Profile and choose one!")

            post = st.text_input("Let's share my current mood as a post!", max_chars=100)
            add_post = st.button('Share Posts')
            if add_post:
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                post = {'Post:': post,
                        'Timestamp': dt_string}
                results = db.child(user['localId']).child("Posts").push(post)
                st.balloons()

            # Display stored summaries and predictions
            summaries = db.child(user['localId']).child("Summaries").get()
            if summaries.val() is not None:
                st.subheader("Stored Summaries")
                for summary in reversed(summaries.each()):
                    st.markdown("**Timestamp:** " + summary.val()['Timestamp'])
                    st.markdown("**Input Text:**")
                    st.info(summary.val()['Text'])
                    st.markdown("**Summary:**")
                    st.success(summary.val()['Summary'])
                    # Display image if available
                    if 'Image' in summary.val():
                        img_str = summary.val()['Image']
                        if img_str is not None:
                            img_data = base64.b64decode(img_str)
                            st.image(img_data, caption='Uploaded Image', use_column_width=True)

                    # Emotion Prediction with Emoji
                    prediction = predict_emotions(summary.val()['Text'])
                    emoji_icon = emotions_emoji_dict.get(prediction, "❓")
                    st.write("**Emotion Prediction:** {} {}".format(prediction, emoji_icon))

                    # Prediction Probability Graph
                    st.success("Prediction Probability")
                    probability = get_prediction_proba(summary.val()['Text'])
                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions", "probability"]
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                    st.altair_chart(fig, use_container_width=True)

                    st.write("---")

            emotions = db.child(user['localId']).child("Emotions").get()
            if emotions.val() is not None:
                st.subheader("Stored Emotion Predictions")
                for emotion in reversed(emotions.each()):
                    st.markdown("**Timestamp:** " + emotion.val()['Timestamp'])
                    st.markdown("**Input Text:**")
                    st.info(emotion.val()['Text'])
                    st.markdown("**Prediction:**")
                    st.write(emotion.val()['Prediction'])
                    st.write("---")
            with col2:
                all_posts = db.child(user['localId']).child("Posts").get()
                if all_posts.val() is not None:
                    for Posts in reversed(all_posts.each()):
                        st.code(Posts.val(), language='')

            # SUMMARIZE TEXT
        elif bio == 'Summarize Text':
            # Your code for summarizing text here
            st.subheader("Summarize Text using txtai")
            input_text = st.text_area("Enter your text here")

            # Image upload option
            uploaded_image = st.file_uploader("Upload an image")

            if st.button("Summarize Text"):
                # Process text and image
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.markdown("**Your Input Text**")
                    st.info(input_text)
                with col2:
                    st.markdown("**Summary Result**")
                    result = text_summary(input_text)
                    st.success(result)
                with col3:
                    st.markdown("**Emotion Prediction**")
                    prediction = predict_emotions(input_text)
                    probability = get_prediction_proba(input_text)
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.write("{}:{}".format(prediction, emoji_icon))
                    st.write("Confidence:{}".format(np.max(probability)))

                    st.success("Prediction Probability")
                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions", "probability"]
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                    st.altair_chart(fig, use_container_width=True)

                    if prediction == "joy":
                        st.info(
                            "Celebrate the moments of joy, for they are the fuel that propels you forward on your journey. Let the warmth of joy fill your heart and illuminate your path. Use this positive energy to inspire others, spread kindness, and create more moments of joy in your life and the lives of those around you. Embrace the beauty of joy, for it is a precious gift that enriches your soul and nourishes your spirit.")
                    elif prediction == "surprise":
                        st.info(
                            "Embrace the magic of surprise as a reminder of life's infinite possibilities. Allow yourself to be swept away by the unexpected, for it is in these moments that we find joy, growth, and new beginnings. Embrace the unknown with open arms, for within it lies the potential for extraordinary experiences and profound transformations. Embrace surprise as a companion on your journey, guiding you to new horizons and unveiling the wonders that await you.")
                    elif prediction == "sadness":
                        st.info(
                            "Sadness is a gentle reminder of our capacity to feel deeply. Allow yourself to acknowledge and honor your emotions, for they are a testament to your humanity. Through moments of sadness, we discover empathy, compassion, and resilience. Remember, just as the clouds pass, so too shall sadness. Embrace the journey, knowing that every tear holds the promise of healing and growth.")
                    elif prediction == "neutral":
                        st.info(
                            "In the calm waters of neutrality, lies the canvas of possibility. Use this moment to reflect, to pause, and to appreciate the beauty of simplicity. In neutrality, there is freedom to explore, to dream, and to discover new paths. Embrace the serenity of the present moment, for within it, lies the potential to shape your future with clarity and purpose.")
                    elif prediction == "anger":
                        st.info(
                            "Harness the power of your anger as fuel for positive change. Use its intensity to drive you towards constructive action and meaningful transformation. Channel your anger into passion for justice, determination for personal growth, and empathy for understanding. Remember, beneath the surface of anger lies a wellspring of energy waiting to be directed towards creating a better world and a better you. Embrace this emotion as a catalyst for empowerment and renewal.")
                    elif prediction == "fear":
                        st.info(
                            "Embrace fear as a catalyst for growth. Recognize it as a sign that you're stepping out of your comfort zone, where real transformation occurs. Channel fear into fuel for courage and resilience. Break free from the chains of apprehension, for within fear lies untapped potential and opportunity. Embrace the unknown with a spirit of curiosity, knowing that overcoming fear leads to personal evolution and empowerment.")
                    elif prediction == "disgust":
                        st.info(
                            "In moments of disgust, remember that discomfort often precedes growth. Use this feeling as a catalyst for change, guiding you toward healthier choices and environments. Embrace the power within you to transform negativity into positivity. Recognize that through confronting what disgusts you, you reclaim control over your surroundings and pave the way for a more fulfilling and authentic life")
                    elif prediction == "shame":
                        st.info(
                            "Shame may weigh heavy, but it doesn't define your worth. Acknowledge it as a signal for growth, not a sentence of condemnation. Embrace vulnerability, for it's the gateway to self-compassion and healing. Recognize that imperfection is part of being human. Let go of shame's grip, and allow self-acceptance to guide you towards a path of authenticity and empowerment.")

                # Store text, image, summary, and prediction in the database
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                # Convert image bytes to base64 string for storage
                if uploaded_image is not None:
                    img_str = base64.b64encode(uploaded_image.read()).decode("utf-8")
                else:
                    img_str = None
                db.child(user['localId']).child("Summaries").push({
                    'Text': input_text,
                    'Summary': result,
                    'Image': img_str,  # Store base64 image string here
                    'Timestamp': dt_string
                })


        # SUMMARIZE DOCUMENT
        elif bio == 'Summarize Document':
            # Your code for summarizing document here
            st.subheader("Summarize Document using txtai")

            input_file = st.file_uploader("Upload your document here", type=['pdf'])

            # Image upload option
            uploaded_image = st.file_uploader("Upload an image")

            if st.button("Summarize Document"):
                with open("doc_file.pdf", "wb") as f:
                    f.write(input_file.getbuffer())

                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    st.info("File uploaded successfully")

                    extracted_text = extract_text_from_pdf("doc_file.pdf")

                    st.markdown("**Extracted Text is Below:**")

                    st.info(extracted_text)

                with col2:
                    st.markdown("**Summary Result**")

                    doc_summary = text_summary(extracted_text)

                    st.success(doc_summary)

                with col3:
                    st.markdown("**Emotion Prediction**")

                    prediction = predict_emotions(extracted_text)

                    probability = get_prediction_proba(extracted_text)

                    emoji_icon = emotions_emoji_dict[prediction]

                    st.write("{}:{}".format(prediction, emoji_icon))

                    st.write("Confidence:{}".format(np.max(probability)))

                    st.success("Prediction Probability")

                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)

                    proba_df_clean = proba_df.T.reset_index()

                    proba_df_clean.columns = ["emotions", "probability"]

                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')

                    st.altair_chart(fig, use_container_width=True)

                    if prediction == "joy":
                        st.info(
                            "Celebrate the moments of joy, for they are the fuel that propels you forward on your journey. Let the warmth of joy fill your heart and illuminate your path. Use this positive energy to inspire others, spread kindness, and create more moments of joy in your life and the lives of those around you. Embrace the beauty of joy, for it is a precious gift that enriches your soul and nourishes your spirit.")
                    elif prediction == "surprise":
                        st.info(
                            "Embrace the magic of surprise as a reminder of life's infinite possibilities. Allow yourself to be swept away by the unexpected, for it is in these moments that we find joy, growth, and new beginnings. Embrace the unknown with open arms, for within it lies the potential for extraordinary experiences and profound transformations. Embrace surprise as a companion on your journey, guiding you to new horizons and unveiling the wonders that await you.")
                    elif prediction == "sadness":
                        st.info(
                            "Sadness is a gentle reminder of our capacity to feel deeply. Allow yourself to acknowledge and honor your emotions, for they are a testament to your humanity. Through moments of sadness, we discover empathy, compassion, and resilience. Remember, just as the clouds pass, so too shall sadness. Embrace the journey, knowing that every tear holds the promise of healing and growth.")
                    elif prediction == "neutral":
                        st.info(
                            "In the calm waters of neutrality, lies the canvas of possibility. Use this moment to reflect, to pause, and to appreciate the beauty of simplicity. In neutrality, there is freedom to explore, to dream, and to discover new paths. Embrace the serenity of the present moment, for within it, lies the potential to shape your future with clarity and purpose.")
                    elif prediction == "anger":
                        st.info(
                            "Harness the power of your anger as fuel for positive change. Use its intensity to drive you towards constructive action and meaningful transformation. Channel your anger into passion for justice, determination for personal growth, and empathy for understanding. Remember, beneath the surface of anger lies a wellspring of energy waiting to be directed towards creating a better world and a better you. Embrace this emotion as a catalyst for empowerment and renewal.")
                    elif prediction == "fear":
                        st.info(
                            "Embrace fear as a catalyst for growth. Recognize it as a sign that you're stepping out of your comfort zone, where real transformation occurs. Channel fear into fuel for courage and resilience. Break free from the chains of apprehension, for within fear lies untapped potential and opportunity. Embrace the unknown with a spirit of curiosity, knowing that overcoming fear leads to personal evolution and empowerment.")
                    elif prediction == "disgust":
                        st.info(
                            "In moments of disgust, remember that discomfort often precedes growth. Use this feeling as a catalyst for change, guiding you toward healthier choices and environments. Embrace the power within you to transform negativity into positivity. Recognize that through confronting what disgusts you, you reclaim control over your surroundings and pave the way for a more fulfilling and authentic life")
                    elif prediction == "shame":
                        st.info(
                            "Shame may weigh heavy, but it doesn't define your worth. Acknowledge it as a signal for growth, not a sentence of condemnation. Embrace vulnerability, for it's the gateway to self-compassion and healing. Recognize that imperfection is part of being human. Let go of shame's grip, and allow self-acceptance to guide you towards a path of authenticity and empowerment.")

                # Store text, image, summary, and prediction in the database
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                # Convert image bytes to base64 string for storage
                if uploaded_image is not None:
                    img_str = base64.b64encode(uploaded_image.read()).decode("utf-8")
                else:
                    img_str = None
                db.child(user['localId']).child("Summaries").push({
                    'Text': extracted_text,
                    'Summary': doc_summary,
                    'Image': img_str,  # Store base64 image string here
                    'Timestamp': dt_string
                })

        # SUMMARIZE AUDI0
        elif bio == 'Summarize Audio':
            # Your code for summarizing audio here
            st.title("Summarize Audio")
            st.write("Select an option to provide audio input.")

            option = st.radio("Select Audio Input Option", ("Upload Audio File", "Record Live Audio"))

            # Image upload option
            uploaded_image = st.file_uploader("Upload an image")

            if option == "Upload Audio File":
                uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

                if uploaded_file:
                    audio_bytes = uploaded_file.read()
                    audio_path = "audio_file.wav"  # Temporary file path for audio file
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)

                    st.audio(audio_path, format="audio/wav")

                    # Button to process the audio
                    if st.button("Summarize Audio"):
                        # Transcribe the audio file
                        aai.settings.api_key = "ed9174d8ec5a45afa0075b544b8eb2d7"
                        transcriber = aai.Transcriber()
                        transcript = transcriber.transcribe(audio_path)

                        # Display the transcription summary
                        st.subheader("Transcription Summary")
                        transcript_summary = text_summary(transcript.text)
                        st.success(transcript_summary)

                        # Display the emotion prediction
                        st.subheader("Emotion Prediction")
                        prediction = predict_emotions(transcript.text)
                        probability = get_prediction_proba(transcript.text)
                        emoji_icon = emotions_emoji_dict[prediction]
                        st.write("{}:{}".format(prediction, emoji_icon))
                        st.write("Confidence:{}".format(np.max(probability)))

                        st.success("Prediction Probability")
                        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                        proba_df_clean = proba_df.T.reset_index()
                        proba_df_clean.columns = ["emotions", "probability"]
                        fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability',
                                                                          color='emotions')
                        st.altair_chart(fig, use_container_width=True)

                        if prediction == "joy":
                            st.info(
                                "Celebrate the moments of joy, for they are the fuel that propels you forward on your journey. Let the warmth of joy fill your heart and illuminate your path. Use this positive energy to inspire others, spread kindness, and create more moments of joy in your life and the lives of those around you. Embrace the beauty of joy, for it is a precious gift that enriches your soul and nourishes your spirit.")
                        elif prediction == "surprise":
                            st.info(
                                "Embrace the magic of surprise as a reminder of life's infinite possibilities. Allow yourself to be swept away by the unexpected, for it is in these moments that we find joy, growth, and new beginnings. Embrace the unknown with open arms, for within it lies the potential for extraordinary experiences and profound transformations. Embrace surprise as a companion on your journey, guiding you to new horizons and unveiling the wonders that await you.")
                        elif prediction == "sadness":
                            st.info(
                                "Sadness is a gentle reminder of our capacity to feel deeply. Allow yourself to acknowledge and honor your emotions, for they are a testament to your humanity. Through moments of sadness, we discover empathy, compassion, and resilience. Remember, just as the clouds pass, so too shall sadness. Embrace the journey, knowing that every tear holds the promise of healing and growth.")
                        elif prediction == "neutral":
                            st.info(
                                "In the calm waters of neutrality, lies the canvas of possibility. Use this moment to reflect, to pause, and to appreciate the beauty of simplicity. In neutrality, there is freedom to explore, to dream, and to discover new paths. Embrace the serenity of the present moment, for within it, lies the potential to shape your future with clarity and purpose.")
                        elif prediction == "anger":
                            st.info(
                                "Harness the power of your anger as fuel for positive change. Use its intensity to drive you towards constructive action and meaningful transformation. Channel your anger into passion for justice, determination for personal growth, and empathy for understanding. Remember, beneath the surface of anger lies a wellspring of energy waiting to be directed towards creating a better world and a better you. Embrace this emotion as a catalyst for empowerment and renewal.")
                        elif prediction == "fear":
                            st.info(
                                "Embrace fear as a catalyst for growth. Recognize it as a sign that you're stepping out of your comfort zone, where real transformation occurs. Channel fear into fuel for courage and resilience. Break free from the chains of apprehension, for within fear lies untapped potential and opportunity. Embrace the unknown with a spirit of curiosity, knowing that overcoming fear leads to personal evolution and empowerment.")
                        elif prediction == "disgust":
                            st.info(
                                "In moments of disgust, remember that discomfort often precedes growth. Use this feeling as a catalyst for change, guiding you toward healthier choices and environments. Embrace the power within you to transform negativity into positivity. Recognize that through confronting what disgusts you, you reclaim control over your surroundings and pave the way for a more fulfilling and authentic life")
                        elif prediction == "shame":
                            st.info(
                                "Shame may weigh heavy, but it doesn't define your worth. Acknowledge it as a signal for growth, not a sentence of condemnation. Embrace vulnerability, for it's the gateway to self-compassion and healing. Recognize that imperfection is part of being human. Let go of shame's grip, and allow self-acceptance to guide you towards a path of authenticity and empowerment.")

                        # Store text, image, summary, and prediction in the database
                        now = datetime.now()
                        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                        # Convert image bytes to base64 string for storage
                        if uploaded_image is not None:
                            img_str = base64.b64encode(uploaded_image.read()).decode("utf-8")
                        else:
                            img_str = None
                        db.child(user['localId']).child("Summaries").push({
                            'Text': transcript.text,
                            'Summary': transcript_summary,
                            'Image': img_str,  # Store base64 image string here
                            'Timestamp': dt_string
                        })

            elif option == "Record Live Audio":
                duration = st.slider("Recording Duration (seconds):", 1, 150, 3)
                fs = 44100  # Sample rate
                channels = 2  # Number of audio channels (1 for mono, 2 for stereo)
                recording = record_audio(duration, fs, channels)

                # Save the recording to a WAV file
                audio_path = "audio_file.wav"
                wavio.write(audio_path, recording, fs, sampwidth=2)

                st.audio(audio_path, format="audio/wav")

                # Button to process the audio
                if st.button("Summarize Audio"):
                    # Transcribe the audio file
                    aai.settings.api_key = "ed9174d8ec5a45afa0075b544b8eb2d7"
                    transcriber = aai.Transcriber()
                    transcript = transcriber.transcribe(audio_path)

                    # Display the transcription summary
                    st.subheader("Transcription Summary")
                    transcript_summary = text_summary(transcript.text)
                    st.success(transcript_summary)

                    # Display the emotion prediction
                    st.subheader("Emotion Prediction")
                    prediction = predict_emotions(transcript.text)
                    probability = get_prediction_proba(transcript.text)
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.write("{}:{}".format(prediction, emoji_icon))
                    st.write("Confidence:{}".format(np.max(probability)))

                    st.success("Prediction Probability")
                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions", "probability"]
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                    st.altair_chart(fig, use_container_width=True)

                    if prediction == "joy":
                        st.info("Celebrate the moments of joy, for they are the fuel that propels you forward on your journey. Let the warmth of joy fill your heart and illuminate your path. Use this positive energy to inspire others, spread kindness, and create more moments of joy in your life and the lives of those around you. Embrace the beauty of joy, for it is a precious gift that enriches your soul and nourishes your spirit.")
                    elif prediction == "surprise":
                        st.info("Embrace the magic of surprise as a reminder of life's infinite possibilities. Allow yourself to be swept away by the unexpected, for it is in these moments that we find joy, growth, and new beginnings. Embrace the unknown with open arms, for within it lies the potential for extraordinary experiences and profound transformations. Embrace surprise as a companion on your journey, guiding you to new horizons and unveiling the wonders that await you.")
                    elif prediction == "sadness":
                        st.info("Sadness is a gentle reminder of our capacity to feel deeply. Allow yourself to acknowledge and honor your emotions, for they are a testament to your humanity. Through moments of sadness, we discover empathy, compassion, and resilience. Remember, just as the clouds pass, so too shall sadness. Embrace the journey, knowing that every tear holds the promise of healing and growth.")
                    elif prediction == "neutral":
                        st.info("In the calm waters of neutrality, lies the canvas of possibility. Use this moment to reflect, to pause, and to appreciate the beauty of simplicity. In neutrality, there is freedom to explore, to dream, and to discover new paths. Embrace the serenity of the present moment, for within it, lies the potential to shape your future with clarity and purpose.")
                    elif prediction == "anger":
                        st.info("Harness the power of your anger as fuel for positive change. Use its intensity to drive you towards constructive action and meaningful transformation. Channel your anger into passion for justice, determination for personal growth, and empathy for understanding. Remember, beneath the surface of anger lies a wellspring of energy waiting to be directed towards creating a better world and a better you. Embrace this emotion as a catalyst for empowerment and renewal.")
                    elif prediction == "fear":
                        st.info("Embrace fear as a catalyst for growth. Recognize it as a sign that you're stepping out of your comfort zone, where real transformation occurs. Channel fear into fuel for courage and resilience. Break free from the chains of apprehension, for within fear lies untapped potential and opportunity. Embrace the unknown with a spirit of curiosity, knowing that overcoming fear leads to personal evolution and empowerment.")
                    elif prediction == "disgust":
                        st.info("In moments of disgust, remember that discomfort often precedes growth. Use this feeling as a catalyst for change, guiding you toward healthier choices and environments. Embrace the power within you to transform negativity into positivity. Recognize that through confronting what disgusts you, you reclaim control over your surroundings and pave the way for a more fulfilling and authentic life")
                    elif prediction == "shame":
                        st.info("Shame may weigh heavy, but it doesn't define your worth. Acknowledge it as a signal for growth, not a sentence of condemnation. Embrace vulnerability, for it's the gateway to self-compassion and healing. Recognize that imperfection is part of being human. Let go of shame's grip, and allow self-acceptance to guide you towards a path of authenticity and empowerment.")

                    # Store text, image, summary, and prediction in the database
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    # Convert image bytes to base64 string for storage
                    if uploaded_image is not None:
                        img_str = base64.b64encode(uploaded_image.read()).decode("utf-8")
                    else:
                        img_str = None
                    db.child(user['localId']).child("Summaries").push({
                        'Text': transcript.text,
                        'Summary': transcript_summary,
                        'Image': img_str,  # Store base64 image string here
                        'Timestamp': dt_string
                    })

