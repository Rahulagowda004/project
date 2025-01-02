from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.googlesearch import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["PHI_API_KEY"] = os.getenv("PHI_API_KEY")

web_search_agent = Agent(
    name = "web search agent",
    role = "Search the web for the information. Do not ask follow up questions",
    model = Groq(id = "llama3-groq-70b-8192-tool-use-preview",api_key=groq_api_key),
    tools = [GoogleSearch()],
    instructions = ["always include sources",
                    "Given a topic by the user, respond with 4 latest news items about that topic.",
        ],
    show_tool_calls=True,
    markdown = True
)

finance_agent = Agent(
    name="financial agent",
    role = "You are trader who has meticulous knowledge of the stock market. You are responsible for providing the user with the financial data of the stock.",
    model = Groq(id = "llama3-groq-70b-8192-tool-use-preview",api_key=groq_api_key),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions=["use tables to display the data", "Analyze the stock using the information provided by the web search agent. Provide relevant financial data, trends, and actionable suggestions for managing the stock, without asking follow-up questions."],
    show_tool_calls = True,
    markdown=True,
)

Multi_ai_agent = Agent(
    model = Groq(id = "llama3-groq-70b-8192-tool-use-preview",api_key=groq_api_key),
    role = "based on data from web search agent and financial agent, provide the user with a comprehensive analysis of the stock. Do not ask follow up questions",
    team = [web_search_agent, finance_agent],
    instructions = ["always include sources"],
    show_tool_calls = True,
    markdown = True
)
Multi_ai_agent.print_response("give me the details of titan company ltd stock",stream=True)