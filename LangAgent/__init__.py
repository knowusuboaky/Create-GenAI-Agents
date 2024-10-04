# LangAgent/__init__.py

# Import agents from the research team
from .research_team.agents import researcher, coder, weather

# Import agents from the logic team
from .logic_team.agents import calculator, reasoner

# Import agents from the analysis team
from .analysis_team.agents import sql_databaser, topic_generator

# Import agents from the reporting team
from .reporting_team.agents import interpreter, summarizer

# Import nodes (supervisor and agent)
from .supervisor import supervisor_chain

# Define what will be accessible when doing `from LLMAgent import *`
__all__ = [
    'researcher', 'coder', 'weather',       # Research team agents
    'calculator', 'reasoner',               # Logic team agents
    'topic_generator', 'sql_databaser',     # Analysis team agents
    'interpreter', 'summarizer',            # Reporting team agents
    'supervisor_chain'                      # Supervisor agent
]
