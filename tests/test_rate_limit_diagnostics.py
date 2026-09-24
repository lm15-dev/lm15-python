"""Shared contract vectors plus real sync/async transport paths; no network."""
import asyncio
import json
import pytest

from conformance.sources import contract_path
from lm15 import Message, OpenAILM, RateLimitError, Request
from lm15.errors import AuthError, ProviderError, map_http_error, with_credential_hint, with_credential_origin
from lm15.providers.async_base import AsyncOpenAILM
from lm15.providers.base import _attach_error_metadata
from lm15.rate_limits import RATE_LIMIT_HEADERS, capture_rate_limits
from lm15.result import materialize_response, amaterialize_response
from lm15.serde import error_detail_from_dict, error_detail_to_dict, stream_event_from_dict, stream_event_to_dict
from lm15.testing import FakeResponse, FakeTransport
from lm15.types import ErrorDetail, StreamErrorEvent
from tests.test_auxiliary_retry_after import CASES, AsyncTransport, make_client

FIXTURE = json.loads(contract_path('errors', 'diagnostic-headers.json').read_text(encoding="utf-8"))
REQUEST = Request(model='deployment', messages=[Message.user('hi')])
HEADERS = [('Retry-After', '39'), ('x-ratelimit-limit-requests', '1'),
           ('x-ratelimit-remaining-requests', '-1'), ('x-ratelimit-reset-requests', '105'),
           ('apim-request-id', 'request-1'), ('api-key', FIXTURE['sentinel'])]


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda c: c['id'])
def test_contract(case):
    error = map_http_error(case['status'], 'provider message', provider='azure',
                           provider_code=case.get('provider_code'),
                           request_id=case.get('body_request_id'), retry_after=case.get('body_retry_after'))
    original = error.message
    _attach_error_metadata(error, case['headers'])
    expect = case['expect']
    assert error.retry_after == expect['retry_after']
    assert error.request_id == expect['request_id']
    assert {k: list(v) for k, v in error.rate_limit_headers.items()} == expect['rate_limit_headers']
    assert str(error) == str(error)  # rendering is repeatable, not appending to message
    assert error.message == original
    assert FIXTURE['sentinel'] not in str(error)
    _attach_error_metadata(error, case['headers'])
    assert str(error).count('Provider rate-limit headers') <= 1


def test_snapshot_is_bounded_immutable_and_copied():
    incoming = {'x-ratelimit-limit-requests': ['1', '2', '3', '4', '5'], 'api-key': [FIXTURE['sentinel']]}
    error = ProviderError('x', rate_limit_headers=incoming)
    incoming['x-ratelimit-limit-requests'][0] = '999'
    assert error.rate_limit_headers['x-ratelimit-limit-requests'] == ('1', '2', '3', '4')
    with pytest.raises(TypeError):
        error.rate_limit_headers['x-ratelimit-limit-requests'] = ('9',)
    for value in ['', '1' * 257, '\t1', '1\n', '\x1b[31m0', '数']:
        assert not capture_rate_limits([('x-ratelimit-limit-requests', value)])
    assert capture_rate_limits([('x-ratelimit-limit-requests', '1' * 256)])
    many = {k: ['1' * 256] * 4 for k in RATE_LIMIT_HEADERS}
    error = ProviderError('original', rate_limit_headers=many)
    assert len(str(error)) < 2300 and 'full retained values' in str(error)
    assert len(error.rate_limit_headers) == len(RATE_LIMIT_HEADERS)


@pytest.mark.parametrize('async_', [False, True])
@pytest.mark.parametrize('operation', ['complete', 'stream', 'list_models'])
def test_http_error_transport_paths(async_, operation):
    response = FakeResponse(status=429, headers=HEADERS,
                            body=b'{"error":{"code":"no_capacity","message":"busy"}}')
    transport = AsyncTransport([response]) if async_ else FakeTransport([response])
    client = (AsyncOpenAILM if async_ else OpenAILM)(api_key='fake', transport=transport)

    async def run():
        if operation == 'stream':
            return [event async for event in client.stream(REQUEST)]
        return await (client.list_models() if operation == 'list_models' else client.complete(REQUEST))

    with pytest.raises(RateLimitError) as caught:
        if async_:
            asyncio.run(run())
        elif operation == 'stream':
            list(client.stream(REQUEST))
        elif operation == 'list_models':
            client.list_models()
        else:
            client.complete(REQUEST)
    error = caught.value
    assert error.rate_limit_headers['x-ratelimit-reset-requests'] == ('105',)
    assert error.retry_after == 39 and error.request_id == 'request-1'
    assert error.provider_code == 'no_capacity' and error.status == 429
    assert len(transport.requests) == 1  # never retry or change the endpoint


@pytest.mark.parametrize('async_', [False, True])
@pytest.mark.parametrize('name,args,kwargs,successful', CASES, ids=[c[0] for c in CASES])
def test_every_auxiliary_path_keeps_limits(async_, name, args, kwargs, successful):
    client, transport = make_client(name, successful, HEADERS, async_)
    with pytest.raises(RateLimitError) as caught:
        result = getattr(client, name)(*args, **kwargs)
        if async_:
            asyncio.run(result)
    assert caught.value.rate_limit_headers['x-ratelimit-remaining-requests'] == ('-1',)
    assert caught.value.request_id == 'request-1'


@pytest.mark.parametrize('async_', [False, True])
def test_http_200_stream_error_and_replay_keep_handshake_diagnostics(async_):
    payload = {'type': 'error', 'error': {'type': 'too_many_requests', 'code': 'no_capacity', 'message': 'busy'}}
    response = FakeResponse(status=200, headers=HEADERS,
                            body=('event: error\ndata: ' + json.dumps(payload) + '\n\n').encode())
    transport = AsyncTransport([response]) if async_ else FakeTransport([response])
    client = (AsyncOpenAILM if async_ else OpenAILM)(api_key='fake', transport=transport)

    async def collect():
        return [event async for event in client.stream(REQUEST)]

    events = asyncio.run(collect()) if async_ else list(client.stream(REQUEST))
    error_event = next(e for e in events if isinstance(e, StreamErrorEvent))
    assert error_event.error.http_response['request_id'] == 'request-1'
    assert error_event.error.provider_code == 'no_capacity'
    assert error_event.error.code == 'rate_limit'
    replay = [stream_event_from_dict(stream_event_to_dict(e)) for e in events]
    assert replay == events
    with pytest.raises(RateLimitError) as caught:
        materialize_response(replay, REQUEST)
    assert caught.value.status is None  # HTTP 200 is not an error's status
    assert caught.value.retry_after == 39 and caught.value.request_id == 'request-1'
    assert caught.value.rate_limit_headers['x-ratelimit-limit-requests'] == ('1',)

    async def source():
        for e in replay:
            yield e
    with pytest.raises(RateLimitError) as caught:
        asyncio.run(amaterialize_response(source(), REQUEST))
    assert caught.value.rate_limit_headers['x-ratelimit-limit-requests'] == ('1',)


@pytest.mark.parametrize('async_', [False, True])
def test_http_200_complete_error_keeps_headers(async_):
    response = FakeResponse(status=200, headers=HEADERS,
                            body=b'{"error":{"code":"no_capacity","message":"busy"}}')
    transport = AsyncTransport([response]) if async_ else FakeTransport([response])
    client = (AsyncOpenAILM if async_ else OpenAILM)(api_key='fake', transport=transport)
    with pytest.raises(RateLimitError) as caught:
        if async_:
            asyncio.run(client.complete(REQUEST))
        else:
            client.complete(REQUEST)
    assert caught.value.request_id == 'request-1'
    assert caught.value.rate_limit_headers['x-ratelimit-limit-requests'] == ('1',)
    assert caught.value.status is None


@pytest.mark.parametrize('async_', [False, True])
def test_stream_parser_raised_exception_keeps_headers(async_):
    class RaisingLM(OpenAILM):
        def parse_stream_events(self, request, raw):
            raise RateLimitError('busy', provider_code='no_capacity')

    response = FakeResponse(status=200, headers=HEADERS, body=b'data: {}\n\n')
    transport = AsyncTransport([response]) if async_ else FakeTransport([response])
    if async_:
        client = AsyncOpenAILM(api_key='fake', transport=transport)
        client._inner.parse_stream_events = RaisingLM.parse_stream_events.__get__(client._inner)
    else:
        client = RaisingLM(api_key='fake', transport=transport)

    async def collect():
        return [e async for e in client.stream(REQUEST)]

    with pytest.raises(RateLimitError) as caught:
        asyncio.run(collect()) if async_ else list(client.stream(REQUEST))
    assert caught.value.rate_limit_headers['x-ratelimit-limit-requests'] == ('1',)
    assert caught.value.request_id == 'request-1'


def test_auth_error_reconstruction_preserves_diagnostics():
    error = AuthError('bad', rate_limit_headers={'retry-after': ['3']})
    assert with_credential_hint(error, 'login').rate_limit_headers == error.rate_limit_headers
    assert with_credential_origin(error, 'explicit').rate_limit_headers == error.rate_limit_headers


def test_canonical_metadata_omission_validation_and_copy():
    assert error_detail_to_dict(ErrorDetail('rate_limit', 'busy')) == {'code': 'rate_limit', 'message': 'busy'}
    raw = {'code': 'rate_limit', 'message': 'busy', 'http_response': {'retry_after': 0, 'rate_limit_headers': {'retry-after': ['3']}}}
    value = error_detail_from_dict(raw)
    raw['http_response']['rate_limit_headers']['retry-after'][0] = '4'
    assert value.http_response['rate_limit_headers']['retry-after'] == ['3']
    assert isinstance(value.http_response['retry_after'], float)
    for bad in [None, [], {'status': 200}, {'retry_after': -1}, {'retry_after': True}, {'retry_after': float('inf')}, {'request_id': ''}, {'rate_limit_headers': {'retry-after': '3'}}]:
        with pytest.raises((TypeError, ValueError)):
            ErrorDetail('rate_limit', 'busy', http_response=bad)
