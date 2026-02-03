"""LangGraph Event Analyzer Agent."""

import re
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from state import AgentState, DEFAULT_SOURCES
from tools.source_analyzer import analyze_source
from tools.event_fetcher import fetch_all_events
from tools.query_parser import match_events_to_query, filter_events_by_llm_result, format_matched_events


def create_agent():
    """Create the LangGraph agent."""

    def source_validation_node(state: AgentState) -> AgentState:
        """Ask user to confirm sources."""
        if state.get("sources_confirmed"):
            return state

        sources_list = "\n".join(f"- {s['name']}" for s in state["sources"])
        message = AIMessage(
            content=f"""I have access to events from these NYC sources:

{sources_list}

Are these sources appropriate for what you're looking for?
- Reply **yes** to proceed with fetching events
- Reply **no** or tell me what other sources you'd like to add"""
        )

        return {
            **state,
            "messages": state["messages"] + [message],
        }

    def route_after_validation(state: AgentState) -> Literal["fetch_events", "add_source", "query_handler", "wait_for_input"]:
        """Route based on user response to source validation."""
        if state.get("sources_confirmed"):
            if state.get("events_fetched"):
                # Check if user wants to refresh
                messages = state.get("messages", [])
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        content = msg.content.lower()
                        if any(word in content for word in ["refresh", "update", "reload", "force"]):
                            return "fetch_events"
                        break
                return "query_handler"
            return "fetch_events"

        # Check last human message
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = msg.content.lower()
                if content in ["yes", "y", "yeah", "yep", "sure", "ok", "okay"]:
                    return "fetch_events"
                elif "http" in content or "www." in content or ".com" in content or ".org" in content:
                    return "add_source"
                elif content in ["no", "n", "nope"]:
                    return "wait_for_input"
                break

        return "wait_for_input"

    def fetch_events_node(state: AgentState) -> AgentState:
        """Fetch events from all sources, using cache when available."""
        force_update = state.get("force_update", False)
        cache_threshold_days = state.get("cache_threshold_days", 7)

        # Check last message for refresh/force keywords
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = msg.content.lower()
                if any(word in content for word in ["refresh", "force", "reload", "update all"]):
                    force_update = True
                break

        if force_update:
            message = AIMessage(content="Force updating events from all sources... This may take a moment.")
        else:
            message = AIMessage(content="Loading events (using cache where available)... This may take a moment.")

        # Fetch the events with cache support
        result = fetch_all_events.invoke({
            "sources": state["sources"],
            "force_update": force_update,
            "cache_threshold_days": cache_threshold_days,
        })

        # Build summary with cache status
        summary_lines = []
        cache_status = result.get("cache_status", {})
        for name, count in result["summary"].items():
            status = cache_status.get(name, "unknown")
            status_emoji = {
                "cached": "📦",
                "fetched": "🔄",
                "stale_cache": "⚠️",
                "error": "❌",
            }.get(status, "")
            summary_lines.append(f"- {name}: {count} events {status_emoji}")

        summary_text = "\n".join(summary_lines)
        summary_text += "\n\n_Legend: 📦=cached, 🔄=freshly fetched, ⚠️=stale cache, ❌=error_"

        if result["errors"]:
            error_text = "\n".join(result["errors"])
            summary_text += f"\n\nErrors encountered:\n{error_text}"

        response = AIMessage(
            content=f"""Done! I found **{result['total']} events** across all sources:

{summary_text}

What would you like to know? You can ask things like:
- "What's good for a date night this weekend?"
- "Something intellectual and fun"
- "Family-friendly activities"
- "**refresh**" to force update all sources"""
        )

        return {
            **state,
            "messages": state["messages"] + [message, response],
            "events": result["events"],
            "sources_confirmed": True,
            "events_fetched": True,
            "force_update": False,
        }

    def add_source_node(state: AgentState) -> AgentState:
        """Handle adding a new source dynamically."""
        messages = state.get("messages", [])
        url = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                urls = re.findall(r'https?://[^\s]+', msg.content)
                if urls:
                    url = urls[0]
                break

        if not url:
            response = AIMessage(
                content="I couldn't find a URL in your message. Please provide the full URL of the event source you'd like to add."
            )
            return {
                **state,
                "messages": state["messages"] + [response],
            }

        print(f"  Analyzing {url}...", flush=True)
        analysis = analyze_source.invoke({"url": url})

        new_source = {
            "name": analysis["name"],
            "url": analysis["url"],
            "method": analysis["method"],
            "enabled": True,
            "default_location": analysis.get("default_location", "See event details"),
            "default_event_type": analysis.get("default_event_type", "Event"),
            "parsing_hints": analysis.get("parsing_hints", ""),
            "wait_seconds": analysis.get("wait_seconds", 3),
            "cloudflare_protected": analysis.get("cloudflare_protected", False),
        }

        if "api_events_path" in analysis:
            new_source["api_events_path"] = analysis["api_events_path"]

        updated_sources = state["sources"] + [new_source]

        response = AIMessage(
            content=f"""I've added a new source:

**{analysis['name']}**
- URL: {url}
- Method: {analysis['method'].upper()}
- Location: {analysis.get('default_location', 'See event details')}
- Event Type: {analysis.get('default_event_type', 'Event')}

{analysis['reason']}

The source has been added. Reply **yes** to fetch events from all sources."""
        )

        return {
            **state,
            "sources": updated_sources,
            "messages": state["messages"] + [response],
            "pending_source": None,
        }

    def query_handler_node(state: AgentState) -> AgentState:
        """Handle user queries about events using semantic matching."""
        messages = state.get("messages", [])
        events = state.get("events", [])

        query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        if not query:
            return state

        if not events:
            response = AIMessage(content="No events loaded. Please fetch events first.")
            return {
                **state,
                "messages": state["messages"] + [response],
            }

        print(f"  Matching query: '{query}'...", flush=True)

        llm_result = match_events_to_query(query, events)
        matched_events = filter_events_by_llm_result(events, llm_result)
        interpretation = llm_result.get("interpretation", "")
        formatted = format_matched_events(matched_events, interpretation)

        response = AIMessage(content=formatted)

        return {
            **state,
            "messages": state["messages"] + [response],
        }

    # Build the graph
    workflow = StateGraph(AgentState)

    workflow.add_node("source_validation", source_validation_node)
    workflow.add_node("fetch_events", fetch_events_node)
    workflow.add_node("add_source", add_source_node)
    workflow.add_node("query_handler", query_handler_node)

    workflow.set_entry_point("source_validation")

    workflow.add_conditional_edges(
        "source_validation",
        route_after_validation,
        {
            "fetch_events": "fetch_events",
            "add_source": "add_source",
            "query_handler": "query_handler",
            "wait_for_input": END,
        },
    )

    workflow.add_edge("fetch_events", END)
    workflow.add_edge("add_source", END)
    workflow.add_edge("query_handler", END)

    return workflow.compile()


def get_initial_state(force_update: bool = False, cache_threshold_days: int = 7) -> AgentState:
    """Get the initial state for the agent."""
    return {
        "sources": DEFAULT_SOURCES,
        "events": [],
        "messages": [],
        "pending_source": None,
        "sources_confirmed": False,
        "events_fetched": False,
        "force_update": force_update,
        "cache_threshold_days": cache_threshold_days,
    }
