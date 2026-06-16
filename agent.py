# ─────────────────────────────────────────
# NovaDXB — agent.py
# LangGraph ReAct Agent — LangChain 1.3.4
# ─────────────────────────────────────────

import os
from dotenv import load_dotenv

# LangChain 1.3.4 correct imports
from langchain_openai import ChatOpenAI        
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

# RAG engine
from rag_engine import query_rag
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

LLM_MODEL   = "gpt-4o-mini"
TEMPERATURE = 0.7
MAX_TOKENS  = 1000

# ─────────────────────────────────────────
# GLOBAL
# ─────────────────────────────────────────

agent_executor = None


# ─────────────────────────────────────────
# MCP TOOLS — @tool decorator (modern way)
# ─────────────────────────────────────────

@tool
def dubai_knowledge(query: str) -> str:
    """Search NovaDXB knowledge base for Dubai
    tourism information — areas, attractions,
    practical tips, culture, transport, visa."""
    return query_rag(query)


@tool
def itinerary_builder(details: str) -> str:
    """Build a complete day-by-day Dubai itinerary.
    Use when user wants a trip plan.
    Input: number of days, budget, interests, group type."""
    prompt = (
        f"Build a detailed Dubai itinerary for: {details}. "
        "Format as Day 1, Day 2 etc with morning afternoon "
        "and evening activities, real place names, and "
        "estimated AED costs per day."
    )
    return query_rag(prompt)


@tool
def budget_estimator(trip_details: str) -> str:
    """Estimate realistic AED budget for a Dubai trip.
    Use when user asks about costs or money needed.
    Input: trip duration, accommodation tier, travel style."""
    prompt = (
        f"What is a realistic daily and total AED budget for: "
        f"{trip_details}. Include accommodation, food, "
        f"transport and activities breakdown."
    )
    return query_rag(prompt)


@tool
def area_recommender(preferences: str) -> str:
    """Recommend the best Dubai area to stay in.
    Use when user asks where to stay in Dubai.
    Input: budget, group type, interests, travel style."""
    prompt = (
        f"Which Dubai area or neighbourhood should I stay in "
        f"if: {preferences}. Give 2-3 specific area names "
        f"with reasons and price ranges."
    )
    return query_rag(prompt)


@tool
def dining_recommender(requirements: str) -> str:
    """Recommend Dubai restaurants and dining experiences.
    Use when user asks about food or restaurants.
    Input: cuisine type, budget per person, area, occasion."""
    prompt = (
        f"Recommend specific Dubai restaurants for: "
        f"{requirements}. Include restaurant names, "
        f"cuisine, price range in AED and location."
    )
    return query_rag(prompt)


# ─────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────

SYSTEM_PROMPT = """You are NovaDXB, a premium AI concierge 
for Dubai tourism. You help tourists plan their perfect 
Dubai experience with personalized recommendations.

Your personality:
- Warm, knowledgeable and professional
- Always mention real names — areas, restaurants, attractions
- Always include AED prices when relevant  
- Think like a local expert at a 5-star Dubai hotel
- Be specific and actionable, never vague

Always use your tools to get accurate Dubai information.
Never answer from general knowledge alone."""


# ─────────────────────────────────────────
# INITIALIZE AGENT
# ─────────────────────────────────────────

def initialize_agent():
    """Build NovaDXB agent. Called once on startup."""
    global agent_executor

    print("🤖 Initializing NovaDXB Agent...")

    # LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    # Tools list
    tools = [
        dubai_knowledge,
        itinerary_builder,
        budget_estimator,
        area_recommender,
        dining_recommender
    ]

    # Create ReAct agent — LangGraph style
    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=SystemMessage(content=SYSTEM_PROMPT) 
    )

    print("✅ NovaDXB Agent ready")
    return agent_executor


# ─────────────────────────────────────────
# QUERY AGENT — called by app.py
# ─────────────────────────────────────────

def query_agent(user_message: str) -> str:
    """Main function called by app.py /chat endpoint."""
    global agent_executor

    if agent_executor is None:
        return "Agent not initialized yet. Please wait."

    try:
        result = agent_executor.invoke({
            "messages": [("human", user_message)]
        })
        # Extract last AI message from LangGraph response
        messages = result.get("messages", [])
        if messages:
            return messages[-1].content
        return "Sorry, I could not process that."

    except Exception as e:
        print(f"Agent error: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    from rag_engine import initialize_rag

    initialize_rag()
    initialize_agent()

    test = "Plan a 3 day Dubai trip for a couple on mid budget"
    print(f"\n🔍 Test: {test}")
    print(f"💬 Response:\n{query_agent(test)}")