# LangAgent/logic_team/agents/calculator.py


# * LIBRARIES

from langchain.agents import AgentExecutor, create_openai_tools_agent,load_tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, Tool
from langchain.chains import LLMMathChain
import os


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

def calculator(llm, system_prompt = None):
    """
    Creates a calculator agent using the provided LLM and search tool.
    
    Args:
    llm: The language model (e.g., ChatOpenAI).
    system_prompt: (Optional) A custom system prompt for the agent. If None, a default prompt is used.
    
    Returns:
    AgentExecutor: The calculator agent ready to answer numeric questions.
    """

    # Default system prompt if none is provided
    if system_prompt is None:
        system_prompt = """
            Use the math_calculator as a mathematics calculator. Useful for when you need to answer numeric questions. 
            Only input math expressions. 
            Example: "What is 2+2?", "Calculate the square root of 16."
        """
            # Create the agent using the helper function
    
    problem_chain = LLMMathChain.from_llm(llm=llm)

    # Create the Math tool
    math_tool = Tool.from_function(
    name="Calculator",
    func=problem_chain.run,
    description="Useful for when you need to answer numeric questions. \
        This tool is only for math questions and nothing else. Only input math expressions, without text."
    )

    calculator_agent_executor = create_agent_with_tools(llm, [math_tool], system_prompt)
    return calculator_agent_executor