"""Exercise the benchmark reader against a real chunked HTTP stream."""
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from bench import common


def event(content):
    body = {"choices": [{"delta": {"content": content}}]}
    return ("data: " + json.dumps(body, ensure_ascii=False) + "\n\n").encode()


END = (b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
       b'"usage":{"prompt_tokens":10,"completion_tokens":4}}\n\n'
       b'data: [DONE]\n\n')


class StreamTiming(unittest.TestCase):
    def request(self, chunks):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for body, delay in chunks:
                    self.wfile.write(f"{len(body):X}\r\n".encode() + body + b"\r\n")
                    self.wfile.flush()
                    time.sleep(delay)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(common, "BASE", f"http://127.0.0.1:{server.server_port}"), \
                 patch.object(common, "API_KEY", ""):
                return common.stream_chat("fixture", [{"role": "user", "content": "test"}],
                                          max_tokens=4, timeout=5)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_arrival_times_survive_small_delayed_chunks(self):
        # The entire response is smaller than read(4096)'s fill target.
        chunks = [(event(str(i)), 0.3) for i in range(4)] + [(END, 0)]
        result = self.request(chunks)
        self.assertIsNone(result["error"], result)
        self.assertEqual(result["text_head"], "0123")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["completion_tokens"], 4)
        self.assertLess(result["ttft_s"], 0.5, result)
        self.assertGreater(result["decode_window_s"], 0.75, result)

    def test_split_utf8_and_coalesced_events_keep_usage_and_text(self):
        body = event("雪")
        split = body.index("雪".encode()) + 1
        chunks = [(b": keepalive\n\n" + body[:split], 0),
                  (body[split:] + event("OK") + END, 0)]
        result = self.request(chunks)
        self.assertIsNone(result["error"], result)
        self.assertEqual(result["text_head"], "雪OK")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["prompt_tokens"], 10)
        self.assertEqual(result["completion_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
