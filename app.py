import asyncio
import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationChain
import os
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from dotenv import load_dotenv
import os


load_dotenv()

st.sidebar.title("Welcome Wanderers", help='This is just a beta model, and is still in progress!!!')
# Add an image to the sidebar
st.sidebar.image("assets/chatbot.jpg") 

st.sidebar.divider()

# Create a sidebar dropdown
selected_option = st.sidebar.selectbox("Select Model:", ["lmsys/fastchat-t5-3b-v1.0", "google/flan-t5-base"])

# Display the selected option below the dropdown
# st.sidebar.write("Model : ", selected_option)

st.sidebar.divider()

max_length = st.sidebar.slider("Max Length", value=132, min_value=32, max_value=250)
temperature = st.sidebar.slider("Temperature", value=0.60, min_value=0.0, max_value=1.0, step=0.05)

repo_id = selected_option
llm = HuggingFaceHub(
    huggingfacehub_api_token=os.getenv('HUGGING_FACE_HUB_API_KEY'),
    repo_id=repo_id,
    model_kwargs={
        'temperature': temperature,
        'max_length': max_length,
    }
)

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=80)
Conversation_buf = ConversationChain(
    llm=llm,
    memory=memory
)

st.markdown("<h1 style='text-align: center;'>Chat Application 🚀🤖</h1>", unsafe_allow_html=True)

st.divider()
default_value = "This is just a small Chat application, and the A.I name is Mr.Zhongli 🤗 This site, \nbuilt by the Me using HuggingFace Models, Its like having a smart machine that \ncompletes your thoughts 😀 Get started by typing a custom snippet, check out the \nrepository, or try one of the examples. Have fun!"
st.text(default_value)

st.divider()

# Create a placeholder for the conversation history
conversation_history_placeholder = st.empty()

# Create a list to store the conversation history
conversation_history = []

user_input = st.text_input("Your Query", max_chars=2024)

if st.button("Predict"):
    # Append user input to the conversation history
    conversation_history.insert(0 ,f"User: {user_input}")
    
    # Await the coroutine to get the actual text
    prediction = asyncio.run(Conversation_buf.acall(inputs=user_input))
    keys_list = list(prediction.items())
    keys = keys_list[2]
    response = keys[1][5:]
    
    # Append model response to the conversation history
    conversation_history.insert(1, f"Mr.Zhongli: {response}")

    # Update the conversation history placeholder
    #conversation_history_placeholder.text_area("Conversation...", "\n".join(conversation_history), height=200)
    st.subheader('_Response_ :blue[here] :sunglasses:')
    st.write(response)
    # st.text(memory.buffer)