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

@tool
def currency_converter(query: str) -> str:
    """Convert an amount between AED (UAE Dirham) and major
    tourist currencies, or explain Dubai money matters.
    Use when user asks about currency conversion, exchange rates,
    or 'how much is X AED in my currency'.
    Input: amount and currency, e.g. '500 AED to USD' or '200 USD to AED'."""
 
    # Fixed approximate rates relative to 1 AED (AED is USD-pegged, very stable)
    rates_per_aed = {
        "USD": 0.272, "EUR": 0.250, "GBP": 0.214, "INR": 22.85,
        "PKR": 75.80, "PHP": 15.40, "CNY": 1.97, "SAR": 1.02,
        "AED": 1.0,
    }
 
    import re as _re
    match = _re.search(
        r"(\d+(?:\.\d+)?)\s*([A-Za-z]{3})\s*(?:to|in)?\s*([A-Za-z]{3})?",
        query, _re.IGNORECASE
    )
 
    if not match:
        return (
            "I can convert between AED and major currencies (USD, EUR, GBP, "
            "INR, PKR, PHP, CNY, SAR). Try asking like '500 AED to USD'."
        )
 
    amount = float(match.group(1))
    from_cur = match.group(2).upper()
    to_cur = (match.group(3) or "AED").upper()
 
    if from_cur not in rates_per_aed or to_cur not in rates_per_aed:
        return (
            f"I support AED conversions with USD, EUR, GBP, INR, PKR, PHP, "
            f"CNY and SAR. I don't have a fixed rate for {from_cur} or {to_cur} — "
            f"please check a live exchange rate for that currency."
        )
 
    # Convert from_cur -> AED -> to_cur
    amount_in_aed = amount / rates_per_aed[from_cur] if from_cur != "AED" else amount
    result = amount_in_aed * rates_per_aed[to_cur]
 
    return (
        f"{amount:.2f} {from_cur} is approximately {result:.2f} {to_cur} "
        f"(AED is pegged to USD at a fixed rate, so this stays very stable). "
        f"Note: exchange houses in Dubai typically offer 3-5% better rates "
        f"than airport counters."
    )
 
 
@tool
def weather_advisor(query: str) -> str:
    """Give weather expectations and best-time-to-visit advice for Dubai.
    Use when user asks about weather, climate, temperature, what to pack,
    or the best month/season to visit.
    Input: a month, season, or general weather question."""
    prompt = (
        f"Based on Dubai's seasonal weather patterns, answer this: {query}. "
        "Include expected temperature range, humidity, and what to pack "
        "if relevant. Mention if it falls in peak, shoulder or low season."
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
    try:
        agent_executor = create_react_agent(
            model=llm,
            tools=tools,
            prompt=SystemMessage(content=SYSTEM_PROMPT)
        )
    except TypeError:
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