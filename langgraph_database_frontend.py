import streamlit as st
from langgraph_database_backend import chatbot, llm, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
import uuid


################## Utility Functions ##################

def generate_thread_id():
    return str(uuid.uuid4())


def get_message_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    return str(content)


def generate_chat_title(user_input):
    try:
        response = llm.invoke(
            [
                HumanMessage(
                    content=(
                        "Create a short, descriptive title for this chat "
                        "based on the user's message. Use 3-6 words maximum. "
                        "Return only the title, without quotation marks.\n\n"
                        f"User message: {user_input}"
                    )
                )
            ]
        )

        title = get_message_text(response.content)

        if title.strip():
            return title.strip().strip('"')[:60]

    except Exception:
        pass

    return " ".join(user_input.split()[:6])[:60] or "New Chat"


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    return state.values.get("messages", [])


def format_messages(messages):
    return [
        {
            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            "content": get_message_text(msg.content)
        }
        for msg in messages
        if isinstance(msg, (HumanMessage, AIMessage))
    ]


def get_thread_title(thread_id):

    if thread_id in st.session_state["chat_titles"]:
        return st.session_state["chat_titles"][thread_id]

    messages = load_conversation(thread_id)

    for msg in messages:
        if isinstance(msg, HumanMessage):

            title = " ".join(
                get_message_text(msg.content).split()[:6]
            )[:60]

            st.session_state["chat_titles"][thread_id] = title

            return title

    return "New Chat"


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []


################## Session Setup ##################

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

if "thread_id" not in st.session_state:

    saved_threads = retrieve_all_threads()

    if saved_threads:

        thread_id = saved_threads[0]

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        st.session_state["message_history"] = format_messages(messages)

    else:

        st.session_state["thread_id"] = generate_thread_id()
        st.session_state["message_history"] = []


if "message_history" not in st.session_state:

    messages = load_conversation(
        st.session_state["thread_id"]
    )

    st.session_state["message_history"] = format_messages(messages)


################## Sidebar ##################

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("＋ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.header("My conversations")


# Retrieve saved thread IDs from SQLite
saved_threads = retrieve_all_threads()

current_thread_id = st.session_state["thread_id"]

if current_thread_id not in saved_threads:
    saved_threads.insert(0, current_thread_id)


for thread_id in saved_threads:

    title = get_thread_title(thread_id)

    if st.sidebar.button(
        title,
        key=f"thread_{thread_id}",
        use_container_width=True
    ):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        st.session_state["message_history"] = format_messages(messages)

        st.rerun()


################## Main Chat UI ##################

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Type your message here...")


if user_input:

    current_thread_id = st.session_state["thread_id"]

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)


    # Generate title from first user message
    if not any(
        message["role"] == "user"
        for message in st.session_state["message_history"][:-1]
    ):

        title = generate_chat_title(user_input)

        st.session_state["chat_titles"][current_thread_id] = title


    config = {
        "configurable": {
            "thread_id": current_thread_id
        }
    }


    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        ai_response = ""

        for message_chunk, metadata in chatbot.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=config,
            stream_mode="messages"
        ):

            if isinstance(message_chunk, (AIMessage, AIMessageChunk)):

                if metadata.get("langgraph_node") != "chat":
                    continue

                content = get_message_text(message_chunk.content)

                if content:

                    ai_response += content

                    response_placeholder.markdown(
                        ai_response + "▌"
                    )

        response_placeholder.markdown(ai_response)


    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )

    st.rerun()