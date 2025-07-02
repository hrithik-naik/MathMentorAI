import streamlit as st
from graph_agent import graph, ChatState

st.set_page_config(page_title="Gemini Chatbot", layout="centered")
st.title("🤖 MathMentorAI (Lang Graph )")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if user_input := st.chat_input("Type your question..."):
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Initialize graph state
    state = ChatState(input=user_input, counter=0)
    ai_response = ""

    try:
        # Stream the LangGraph execution
        for chunk in graph.stream(state):
            for node_name, node_state in chunk.items():
                if node_name == "FeedbackCollector":
                    continue 
                ai_response += f"### 🧠 Node: `{node_name}`\n"
                if hasattr(node_state, 'output'):
                    ai_response += f"{node_state.output}\n\n"
                elif hasattr(node_state, 'content'):
                    ai_response += f"{node_state.content}\n\n"
                elif isinstance(node_state, dict):
                    if 'output' in node_state:
                        ai_response += f"{node_state['output']}\n\n"
                    elif 'content' in node_state:
                        ai_response += f"{node_state['content']}\n\n"
                    else:
                        ai_response += f"{str(node_state)}\n\n"
                else:
                    ai_response += f"{str(node_state)}\n\n"

    except Exception as e:
        try:
            result = graph.invoke(state)
            ai_response = f"### 🤖 Final Response\n"
            if hasattr(result, 'output'):
                ai_response += result.output
            elif hasattr(result, 'content'):
                ai_response += result.content
            else:
                ai_response += str(result)
        except Exception as e2:
            ai_response = f"❌ Error executing graph: {str(e2)}\n\nOriginal streaming error: {str(e)}"

    # Append assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)

    # ✍️ Human-in-the-loop feedback input
    with st.expander("💡 Provide feedback on this response"):
        feedback_text = st.text_area("Your feedback", key=f"feedback_{len(st.session_state.chat_history)}")
        if st.button("Submit Feedback"):
            st.success("✅ Feedback submitted!")
            # Save feedback log
            with open("feedback_log.txt", "a", encoding="utf-8") as f:
                f.write(f"Q: {user_input}\nA: {ai_response}\nFeedback: {feedback_text}\n{'='*80}\n")
