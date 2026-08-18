"""Development-only localhost OpenAI-compatible PAFA proposer stub.

This is not an LLM and must never be used for paper results. It returns one
fixed, schema-valid proposal so the local ClientApp/LLM transport can be
exercised without sending client state to a public endpoint.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _Handler(BaseHTTPRequestHandler):
    server_version = "AQFL-LocalLLMStub/1.0"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        payload: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "proposals": [
                                    {
                                        "client_id": "local",
                                        "diagnosis": "stable",
                                        "evidence": ["development stub response"],
                                        "candidate_action_ids": ["safe_default"],
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11434)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
