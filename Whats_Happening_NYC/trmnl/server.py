"""Simple server for TRMNL plugin."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from event_selector import get_trmnl_events

TEMPLATE_PATH = Path(__file__).parent / "template.html"


def render_template(events: list[dict]) -> str:
    """
    Render the TRMNL template with events.

    Uses simple string replacement for Liquid-style tags.
    """
    template = TEMPLATE_PATH.read_text()

    # Build the events HTML
    event_html_parts = []
    for event in events:
        event_block = f'''
      <div class="flex flex--col gap--xs border rounded p--sm">
        <div class="flex gap--xs">
          <span class="label label--underline">{event["day"]}</span>
          <span class="label">{event["date"]}</span>
        </div>
        <span class="value value--sm">{event["time"]}</span>
        <div class="description description--sm">{event["name"]}</div>
        <span class="label label--xs opacity-50">{event["source"]}</span>
      </div>'''
        event_html_parts.append(event_block)

    events_html = "\n".join(event_html_parts)

    # Replace the template loop with rendered HTML
    import re
    pattern = r'\{% for event in events %\}.*?\{% endfor %\}'
    result = re.sub(pattern, events_html, template, flags=re.DOTALL)

    return result


def get_events_json() -> dict:
    """Get events data as JSON for TRMNL polling."""
    events = get_trmnl_events(count=6, days=6)
    return {"events": events}


class TRMNLHandler(BaseHTTPRequestHandler):
    """HTTP handler for TRMNL requests."""

    def do_GET(self):
        if self.path == "/events.json":
            # Return JSON data for TRMNL polling
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = get_events_json()
            self.wfile.write(json.dumps(data).encode())

        elif self.path == "/" or self.path == "/preview":
            # Return rendered HTML for preview
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            events = get_trmnl_events(count=6, days=6)
            html = render_template(events)
            # Wrap in basic HTML for preview
            preview = f"""<!DOCTYPE html>
<html>
<head>
    <title>TRMNL NYC Events Preview</title>
    <link rel="stylesheet" href="https://usetrmnl.com/css/latest/plugins.css">
    <style>
        body {{ background: #f0f0f0; padding: 20px; }}
        .screen {{
            width: 800px;
            height: 480px;
            background: white;
            margin: 0 auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
{html}
</body>
</html>"""
            self.wfile.write(preview.encode())

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[TRMNL] {args[0]}")


def run_server(port: int = 8080):
    """Run the TRMNL plugin server."""
    server = HTTPServer(("", port), TRMNLHandler)
    print(f"TRMNL NYC Events server running at http://localhost:{port}")
    print(f"  Preview: http://localhost:{port}/preview")
    print(f"  JSON:    http://localhost:{port}/events.json")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
