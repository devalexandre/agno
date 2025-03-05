from agno.agent.agent import Agent
from agno.models.deepseek.deepseek import DeepSeek
from agno.models.openai.chat import OpenAIChat

task = "Write a short story about life in 500000 years"

reasoning_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    reasoning_model=DeepSeek(id="deepseek-reasoner"),
    markdown=True,
)
reasoning_agent.print_response(task, stream=True)
