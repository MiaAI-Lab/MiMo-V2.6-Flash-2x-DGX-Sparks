"""Send the saved synthetic FHD and 4K cards to an already running endpoint."""
import argparse
import base64
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8888')
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parent / 'fixtures' / 'vision'
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('BENCH_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['BENCH_API_KEY']
    prompt = ('Read the printed CARD number and the color of the large rectangle '
              'in each image. Return only a JSON array in image order, each object '
              'with card (integer) and color (lowercase color name).')
    passed = True
    for filename, card, color in [('card-fhd.png', 873, 'green'),
                                  ('card-4k.png', 651, 'yellow')]:
        image = base64.b64encode((fixtures / filename).read_bytes()).decode()
        body = {
            'model': args.model, 'stream': True,
            'stream_options': {'include_usage': True}, 'temperature': 0,
            'chat_template_kwargs': {'enable_thinking': False}, 'max_tokens': 256,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + image}},
            ]}],
        }
        request = Request(args.base_url.rstrip('/') + '/v1/chat/completions',
                          data=json.dumps(body).encode(), headers=headers)
        text, finish, usage, done = '', None, {}, False
        with urlopen(request, timeout=180) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(event['error'])
                usage = event.get('usage') or usage
                for choice in event.get('choices', []):
                    text += (choice.get('delta') or {}).get('content') or ''
                    finish = choice.get('finish_reason') or finish
        content = text.strip()
        if content.startswith('```') and content.endswith('```'):
            content = '\n'.join(content.splitlines()[1:-1])
        try:
            actual = json.loads(content)
        except json.JSONDecodeError:
            actual = None
        expected = [{'card': card, 'color': color}]
        correct = done and finish == 'stop' and actual == expected
        passed &= correct
        print(json.dumps({'image': filename, 'expected': expected, 'response': text,
                          'finish_reason': finish, 'usage': usage, 'passed': correct}),
              flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
