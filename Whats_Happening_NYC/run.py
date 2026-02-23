"""Entry point for running the NYC Event Analyzer Agent."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import HumanMessage

from agent import create_agent, get_initial_state


def main():
    """Run the event analyzer agent interactively."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NYC Event Analyzer Agent")
    parser.add_argument(
        "--force-update", "-f",
        action="store_true",
        help="Force update all sources, bypassing cache"
    )
    parser.add_argument(
        "--cache-days", "-c",
        type=int,
        default=7,
        help="Number of days before cache is considered stale (default: 7)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NYC Event Analyzer Agent")
    print("=" * 60)

    if args.force_update:
        print("Mode: Force update (bypassing cache)")
    else:
        print(f"Mode: Using cache (threshold: {args.cache_days} days)")

    print("\nInitializing agent...\n")

    # Create the agent and get initial state
    agent = create_agent()
    state = get_initial_state(
        force_update=args.force_update,
        cache_threshold_days=args.cache_days,
    )

    # Run the initial source validation
    result = agent.invoke(state)
    state = result

    # Print the agent's message
    for msg in state["messages"]:
        print(msg.content)
        print()

    # Interactive loop
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            # Add user message to state
            prev_msg_count = len(state["messages"])
            state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

            # Run the agent
            result = agent.invoke(state)
            state = result

            # Print new messages (skip the user message we just added)
            new_messages = state["messages"][prev_msg_count + 1:]
            for msg in new_messages:
                if hasattr(msg, "content") and msg.content:
                    print(f"\nAgent: {msg.content}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()
