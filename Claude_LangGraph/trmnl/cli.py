#!/usr/bin/env python3
"""TRMNL NYC Events CLI.

Usage:
    python cli.py preview     - Generate HTML preview
    python cli.py push        - Push events to TRMNL webhook
    python cli.py serve       - Run local preview server
    python cli.py json        - Output events as JSON
    python cli.py markup      - Print the Liquid markup template
"""

import argparse
import json
import sys
import os
from pathlib import Path

from event_selector import get_trmnl_events
from trmnl_push import push_to_trmnl, get_markup


def cmd_preview(args):
    """Generate HTML preview file."""
    events = get_trmnl_events(count=args.count, days=args.days)

    if not events:
        print("No events found.")
        return 1

    # Build event cards HTML using TRMNL classes
    cards = []
    for e in events:
        if e["time"]:
            time_html = f'<span class="value value--small">{e["time"]}</span>'
        else:
            time_html = '<span class="label label--small">See site</span>'

        desc_html = f'<span class="label label--small">{e["description"]}</span>' if e.get("description") else ''

        card = f'''        <div class="item col--span-1">
          <div class="content">
            <div class="row">
              <span class="label label--underline">{e["day"]}</span>
              <span class="label">{e["date"]}</span>
              {time_html}
            </div>
            <span class="title title--small">{e["name"]}</span>
            <div class="divider"></div>
            {desc_html}
            <span class="label label--small label--gray">{e["source"]}</span>
          </div>
        </div>'''
        cards.append(card)

    events_html = "\n".join(cards)

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>TRMNL NYC Events Preview</title>
    <link rel="stylesheet" href="https://trmnl.com/css/latest/plugins.css">
    <style>
        body {{
            background: #e0e0e0;
            padding: 40px;
            display: flex;
            justify-content: center;
        }}
        .screen {{
            width: 800px;
            height: 480px;
            background: white;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            overflow: hidden;
        }}
    </style>
</head>
<body>
<div class="screen">
  <div class="layout layout--col gap--space-between">
    <div class="grid grid--cols-3">
      <div class="row">
{events_html}
      </div>
    </div>
  </div>
  <div class="title_bar">
    <img class="image" src="https://usetrmnl.com/images/plugins/trmnl--render.svg" />
    <span class="title">NYC Events</span>
  </div>
</div>
</body>
</html>'''

    output_path = Path(args.output) if args.output else Path(__file__).parent / "preview.html"
    output_path.write_text(html)
    print(f"Preview saved to: {output_path}")
    print(f"Open in browser: file://{output_path.absolute()}")

    return 0


def cmd_push(args):
    """Push events to TRMNL webhook."""
    url = args.url or os.environ.get("TRMNL_WEBHOOK_URL")

    if not url:
        print("Error: No webhook URL provided.")
        print("Use --url or set TRMNL_WEBHOOK_URL environment variable.")
        return 1

    result = push_to_trmnl(url)

    if result["success"]:
        print(f"Success: {result['message']}")
        for i, e in enumerate(result["events"], 1):
            print(f"  {i}. {e['day']} {e['date']} - {e['name'][:40]}")
        return 0
    else:
        print(f"Error: {result['error']}")
        return 1


def cmd_serve(args):
    """Run local preview server."""
    from server import run_server
    run_server(port=args.port)


def cmd_json(args):
    """Output events as JSON."""
    events = get_trmnl_events(count=args.count, days=args.days)
    output = {"events": events}
    print(json.dumps(output, indent=2))
    return 0


def cmd_markup(args):
    """Print the Liquid markup template."""
    print(get_markup())
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="TRMNL NYC Events - Display random NYC events on your TRMNL device",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py preview              Generate HTML preview
  python cli.py push --url <url>     Push to TRMNL webhook
  python cli.py serve --port 8080    Run preview server
  python cli.py json --count 3       Output 3 events as JSON

Setup:
  1. Create a Private Plugin in TRMNL with "Webhook" strategy
  2. Copy the Webhook URL from your plugin settings
  3. Paste the markup from 'python cli.py markup' into your plugin's Markup Editor
  4. Run 'python cli.py push --url <your-webhook-url>' to send events
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Preview command
    p_preview = subparsers.add_parser("preview", help="Generate HTML preview file")
    p_preview.add_argument("-o", "--output", help="Output file path")
    p_preview.add_argument("-c", "--count", type=int, default=6, help="Number of events (default: 6)")
    p_preview.add_argument("-d", "--days", type=int, default=6, help="Days to look ahead (default: 6)")
    p_preview.set_defaults(func=cmd_preview)

    # Push command
    p_push = subparsers.add_parser("push", help="Push events to TRMNL webhook")
    p_push.add_argument("-u", "--url", help="TRMNL webhook URL (or set TRMNL_WEBHOOK_URL)")
    p_push.set_defaults(func=cmd_push)

    # Serve command
    p_serve = subparsers.add_parser("serve", help="Run local preview server")
    p_serve.add_argument("-p", "--port", type=int, default=8080, help="Port (default: 8080)")
    p_serve.set_defaults(func=cmd_serve)

    # JSON command
    p_json = subparsers.add_parser("json", help="Output events as JSON")
    p_json.add_argument("-c", "--count", type=int, default=6, help="Number of events (default: 6)")
    p_json.add_argument("-d", "--days", type=int, default=6, help="Days to look ahead (default: 6)")
    p_json.set_defaults(func=cmd_json)

    # Markup command
    p_markup = subparsers.add_parser("markup", help="Print the Liquid markup template")
    p_markup.set_defaults(func=cmd_markup)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
