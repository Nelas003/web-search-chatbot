import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

from agents import Agent, Runner, function_tool, set_tracing_disabled
from openai import AsyncOpenAI
from duckduckgo_search import DDGS

# --- Load environment variables ---
load_dotenv()
token = os.getenv("OPENAI_API_KEY")
if not token:
    st.error("SECRET (API key) not found. Please set it in your .env file.")
    st.stop()

# --- Set up OpenAI Agent SDK ---
set_tracing_disabled(True)

client = AsyncOpenAI(
    base_url="https://api.openai.com/v1",  # or github endpoint if using github model
    api_key=token,
)

from agents import OpenAIChatCompletionsModel, ModelSettings

model_instance = OpenAIChatCompletionsModel(
    model="gpt-4.1-nano",  # or your preferred model e.g., "gpt-4o"
    openai_client=client,
)

# --- DuckDuckGo Search Tool ---
@function_tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return f"No search results found for: {query}"
            summary = f"Search results for '{query}':\n\n"
            for i, r in enumerate(results, 1):
                summary += f"{i}. **{r['title']}**\n   {r['body']}\n   Source: {r['href']}\n\n"
            return summary
    except Exception as e:
        return f"Error during search: {e}"

# --- Agent Setup ---
agent = Agent(
    name="WebGPT",
    instructions="""
    You are a helpful assistant. Use the web_search tool if a question requires current or external information.
    Cite your sources when using search results.
    """,
    model=model_instance,
    model_settings=ModelSettings(temperature=0.1),
    tools=[web_search]
)

# --- Async Handler ---
async def get_agent_reply(prompt: str) -> str:
    result = await Runner.run(agent, prompt)
    return result.final_output or "No response."

# --- Streamlit UI ---
st.title("Smart Agent Chatbot")
st.write("Ask anything! The agent can also search the web if it doesn't know something.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display existing messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New user prompt
if user_prompt := st.chat_input("Ask me something..."):
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(get_agent_reply(user_prompt))
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

# Sidebar tools
with st.sidebar:
    st.header("Chat Options")
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("Web search with DuckDuckGo")
    st.markdown("OpenAI Agent SDK")
    st.markdown("Uses function tools to retrieve current info")
