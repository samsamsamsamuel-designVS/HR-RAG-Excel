"""Chat Completions gateway adapter for the existing RAG response contract.

Requires an OpenAI-compatible gateway; the model alias does not imply that
Azure's native API or Anthropic's Messages endpoint implements this protocol.
"""
import json
from types import SimpleNamespace
from urllib.parse import urlparse

from openai import OpenAI, omit


class ModelClient:
    def __init__(self, *, api_key, base_url, timeout, max_retries=1):
        url = urlparse(base_url)
        if url.scheme not in ('http', 'https') or not url.netloc:
            raise ValueError('Set AI_BASE_URL in .env to your AI platform API base URL.')
        if not api_key:
            raise ValueError('Set AI_API_KEY in .env and restart Streamlit.')
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             timeout=timeout, max_retries=max_retries,
                             default_headers={'x-api-key': api_key})
        self.responses = self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def create(self, *, model, instructions, input, max_output_tokens,
               store=False, text=None):
        # Ask for JSON in the prompt for gateways without strict response_format
        # support. Pydantic validation and the RAG retry remain mandatory.
        if text:
            schema = text['format']['schema']
            instructions += ('\nReturn ONLY a JSON object matching this schema. '
                             'Do not use Markdown fences or explanatory text.\n'
                             + json.dumps(schema))
        result = self.client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': instructions},
                      {'role': 'user', 'content': input}],
            max_tokens=max_output_tokens,
            extra_headers={"Authorization": omit},
        )
        if not result.choices:
            raise ValueError('The AI platform returned no response choices.')
        choice = result.choices[0]
        refusal = getattr(choice.message, 'refusal', None)
        return SimpleNamespace(
            status='completed' if choice.finish_reason == 'stop' and not refusal else 'incomplete',
            output_text=choice.message.content or '',
            output=[SimpleNamespace(content=[SimpleNamespace(type='refusal')])] if refusal else [],
            incomplete_details=SimpleNamespace(
                reason='max_output_tokens' if choice.finish_reason == 'length' else choice.finish_reason),
        )
