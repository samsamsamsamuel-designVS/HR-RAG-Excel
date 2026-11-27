"""Verify gateway routing and response handling without network access."""
import json
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

from model_client import ModelClient
from config import load_settings
from rag import request_query_plan


class GatewayTests(unittest.TestCase):
    def client(self, responses, requests):
        def handle(request):
            requests.append(request)
            return httpx.Response(200, json=responses.pop(0))
        def make_sdk(**kwargs):
            return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        with patch('model_client.OpenAI', side_effect=make_sdk):
            return ModelClient(api_key='gateway-test-key',
                               base_url='https://api.ai.tech.gov.sg/platform/models/v1', timeout=60)

    def response(self, content, finish='stop'):
        return {'id': 'test', 'object': 'chat.completion', 'created': 0,
                'model': 'gpt-5.6-sol', 'choices': [
                    {'index': 0, 'finish_reason': finish,
                     'message': {'role': 'assistant', 'content': content}}]}

    def test_gateway_request_and_validated_plan(self):
        requests = []
        with self.client([self.response('{"sql":"SELECT 1","clarification":""}')], requests) as client:
            plan = request_query_plan(client, load_settings(), instructions='Plan', input='Question')
        self.assertEqual(plan.sql, 'SELECT 1')
        self.assertEqual(str(requests[0].url), 'https://api.ai.tech.gov.sg/platform/models/v1/chat/completions')
        self.assertEqual(requests[0].headers['x-api-key'], 'gateway-test-key')
        self.assertNotIn('authorization', requests[0].headers)
        body = json.loads(requests[0].content)
        self.assertEqual(body['model'], 'gpt-5.6-sol')
        self.assertIn('schema', body['messages'][0]['content'])
        self.assertNotIn('store', body)
        self.assertNotIn('response_format', body)

    def test_truncation_retries_and_validates(self):
        requests = []
        responses = [self.response('{', 'length'),
                     self.response('{"sql":"SELECT 2","clarification":""}')]
        with self.client(responses, requests) as client:
            plan = request_query_plan(client, load_settings(), instructions='Plan', input='Question')
        self.assertEqual(plan.sql, 'SELECT 2')
        self.assertGreater(json.loads(requests[1].content)['max_tokens'],
                           json.loads(requests[0].content)['max_tokens'])

    def test_missing_endpoint_never_defaults_to_openai(self):
        with patch('model_client.OpenAI') as sdk:
            with self.assertRaisesRegex(ValueError, 'AI_BASE_URL'):
                ModelClient(api_key='test', base_url='', timeout=60)
            sdk.assert_not_called()

    def test_content_filter_not_completed(self):
        with self.client([self.response('', 'content_filter')], []) as client:
            response = client.create(model='gpt-5.6-sol', instructions='Answer',
                                     input='Question', max_output_tokens=100)
        self.assertEqual(response.status, 'incomplete')
