from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import phi.api
from phi.playground import Playground,serve_playground_app
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
phi.api = os.getenv("PHI_API_KEY")

web_search_agent = Agent(
    name = "web search agent",
    role = "Search the web for the information",
    model = Groq(id = "llama3-groq-70b-8192-tool-use-preview",api_key=groq_api_key),
    tools = [DuckDuckGo()],
    instructions = ["always include sources"],
    show_tool_calls=True,
    markdown = True
)

finance_agent = Agent(
    name="financial agent",
    model = Groq(id = "llama3-groq-70b-8192-tool-use-preview",api_key=groq_api_key),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions=["use tables to display the data"],
    show_tool_calls = True,
    markdown=True
)

app = Playground(agents = [finance_agent, web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app",reload = True)