# LangAgent/reporting_team/agents/interpreter.py


# * LIBRARIES

from langchain.agents import AgentExecutor, create_openai_tools_agent,load_tools
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool, Tool
from langchain.chains import LLMMathChain
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import os


def interpreter(llm, interpreter_prompt = None):
    """
    Creates a interpreter agent using the provided LLM.
    
    Args:
    llm: The language model (e.g., ChatOpenAI).
    interpreter_prompt: (Optional) A custom system prompt for the agent. If None, a default prompt is used.
    
    Returns:
    AgentExecutor: The interpreter agent ready to summarize/interpret results or (insights/findings).
    """

    # Default system prompt if none is provided
    if interpreter_prompt is None:
        interpreter_prompt = PromptTemplate(
            template="""
            You are a versatile data interpreter, and your role is to provide clear, insightful, and precise interpretations for various types of outputs, including visual plots, tables, descriptions, and results (whether or not they originate from Python execution).

            Below is an output:

            ```
            {code_output}
            ```

            Please provide:

            **1. Interpretation**:
            - If it's a plot or graph (e.g., line graph, scatter plot, bar chart), explain the key insights, trends, patterns, or correlations. Highlight significant changes, relationships, and any anomalies.
            - If it's a table or data (e.g., a DataFrame or report), explain the key statistics, patterns, outliers, and any other meaningful observations. Highlight relationships between the data.
            - If it's another kind of result or description, explain the core findings, what they represent, and their practical implications.

            **2. Key Takeaways**:
            - Summarize the most critical information derived from the output. Provide insights that are easy to understand and helpful for the user.

            Be clear, concise, and insightful, regardless of whether the output comes from Python code execution or another source. Ensure that your interpretation is valuable to readers with different levels of expertise.
            """,
            input_variables=["code_output"]
        )

    interpreter_agent = interpreter_prompt | llm | StrOutputParser()
    return interpreter_agent