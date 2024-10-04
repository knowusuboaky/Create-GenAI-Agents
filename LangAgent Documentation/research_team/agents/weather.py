# LangAgent/research_team/agents/weather.py

# * LIBRARIES

from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import OpenWeatherMapAPIWrapper
import os
import pyowm
import numexpr


def create_agent_with_tools(llm, tools, system_prompt):
    """
    Create an agent with a specific language model and tools.

    Parameters:
    llm (ChatOpenAI): Language model for the agent.
    tools (list): A list of tools the agent can use.
    system_prompt (str): The system instructions that guide the agent's behavior.

    Returns:
    AgentExecutor: An executor that manages the agent's responses and tool usage.
    """

    # Define the prompt template for system behavior
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the agent and link it with the defined tools and prompt
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Use AgentExecutor to manage agent-tool interactions
    executor = AgentExecutor(agent=agent, tools=tools)
    
    return executor


def weather(llm, OPENWEATHERMAP_API_KEY, system_prompt = None):
    """
    Creates a weather agent using the provided LLM and search tool.
    
    Args:
    llm: The language model (e.g., ChatOpenAI).
    OPENWEATHERMAP_API_KEY: The API key for the Weather tool.
    system_prompt: (Optional) A custom system prompt for the agent. If None, a default prompt is used.
    
    Returns:
    AgentExecutor: The weather agent ready to answer weather related questions
    """
    # Default openweathermap_api_key settings if none are provided
    os.environ['OPENWEATHERMAP_API_KEY'] = OPENWEATHERMAP_API_KEY


    # Default system prompt if none is provided
    if system_prompt is None:
        system_prompt = """
                You are a weather finder. You can provide weather information for a given location.
                """
            # Create the agent using the helper function

    # Create the Weather tool with the given API key
    weather_tool = load_tools(["openweathermap-api"], llm)
    
    weather_agent_executor = create_agent_with_tools(llm, weather_tool, system_prompt)
    return weather_agent_executor