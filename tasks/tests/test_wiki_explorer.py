"""A failed Wikipedia search is reported, not passed off as a page that is absent.

The distinction decides whether a run's results mean anything. A search that
cannot reach Wikipedia and one for a page that does not exist both leave the
agent with nothing, so if the two are reported alike a run against a blocked
endpoint scores every episode zero and reads as a weak agent rather than a
broken experiment.
"""

from types import SimpleNamespace

import wikipedia

import pytest

from tasks.envs import utils
from tasks.envs.utils import LangChainWiki, WikipediaUnavailable


@pytest.fixture(autouse=True)
def waits(monkeypatch):
    """The seconds the retry would have slept, without spending them."""
    slept = []
    monkeypatch.setattr(utils.time, 'sleep', slept.append)
    return slept


class Throttled:
    """Wikimedia's 429: an HTML error page and the wait it asks for."""

    def __init__(self):
        self.status_code = 429
        self.headers = {'Retry-After': '11', 'Content-Type': 'text/html; charset=utf-8'}
        self.text = '<!DOCTYPE html><title>Wikimedia Error</title>'

    def json(self):
        raise ValueError('Expecting value: line 1 column 1 (char 0)')


@pytest.fixture
def explorer(monkeypatch):
    return LangChainWiki()


def raising(*failures):
    """A `wikipedia.page` that raises `failures` in turn, then returns a page."""
    attempts = []

    class Page:
        content = 'the page text'
        url = 'https://en.wikipedia.org/wiki/Telemundo'

    def page(title, **kwargs):
        attempts.append(title)
        if len(attempts) <= len(failures):
            raise failures[len(attempts) - 1]
        return Page()

    return page, attempts


def test_a_page_that_does_not_exist_is_reported_as_absent(explorer, monkeypatch):
    monkeypatch.setattr(wikipedia, 'page', lambda title, **kwargs: (_ for _ in ()).throw(
        wikipedia.PageError('no such page')))
    monkeypatch.setattr(wikipedia, 'search', lambda title, **kwargs: ['Something else'])

    result = explorer.search('Nonexistent Title')

    assert 'Could not find [Nonexistent Title]' in result
    assert 'Something else' in result, 'the similar titles are what the agent searches next'


def test_a_disambiguation_is_reported_as_absent_with_the_alternatives(explorer, monkeypatch):
    monkeypatch.setattr(wikipedia, 'page', lambda title, **kwargs: (_ for _ in ()).throw(
        wikipedia.DisambiguationError('ambiguous', [])))
    monkeypatch.setattr(wikipedia, 'search', lambda title, **kwargs: ['Mercury (planet)'])

    result = explorer.search('Mercury')

    assert 'Could not find [Mercury]' in result
    assert 'Mercury (planet)' in result


def test_being_unable_to_reach_wikipedia_is_raised_rather_than_reported_as_absent(
    explorer, monkeypatch
):
    """This is the one that was silent.

    Wikimedia answers the `wikipedia` package's shared default User-Agent with
    429 and a text/plain body, which the package fails to parse as JSON. That was
    caught alongside PageError and turned into `Could not find`, so every search
    of a run came back empty and the results looked like a weak agent.
    """
    monkeypatch.setattr(wikipedia, 'page', lambda title, **kwargs: (_ for _ in ()).throw(
        ValueError('Expecting value: line 1 column 1 (char 0)')))

    with pytest.raises(WikipediaUnavailable, match='could not be reached'):
        explorer.search('Telemundo')


def test_the_transport_failure_names_what_went_wrong(explorer, monkeypatch):
    """The message is the only place the cause survives, so it carries it."""
    monkeypatch.setattr(wikipedia, 'page', lambda title, **kwargs: (_ for _ in ()).throw(
        ValueError('Expecting value: line 1 column 1 (char 0)')))

    with pytest.raises(WikipediaUnavailable) as failure:
        explorer.search('Telemundo')

    assert 'Telemundo' in str(failure.value)
    assert 'ValueError' in str(failure.value)
    assert isinstance(failure.value.__cause__, ValueError)


def test_a_page_is_fetched_once_rather_than_twice(explorer, monkeypatch):
    """It used to ask for `.content` and `.url` off two separate fetches, which is
    two round trips per search and twice the rate limit spent."""
    calls = []

    class Page:
        content = 'the page text'
        url = 'https://en.wikipedia.org/wiki/Telemundo'

    def page(title, **kwargs):
        calls.append(title)
        return Page()

    monkeypatch.setattr(wikipedia, 'page', page)
    explorer.search('Telemundo')

    assert calls == ['Telemundo'], f'fetched {len(calls)} times'


def test_a_search_that_succeeds_keeps_the_page_for_lookup(explorer, monkeypatch):
    class Page:
        content = 'First paragraph.\n\nSecond paragraph mentioning zebras.'
        url = 'https://en.wikipedia.org/wiki/Telemundo'

    monkeypatch.setattr(wikipedia, 'page', lambda title, **kwargs: Page())
    explorer.search('Telemundo')

    assert 'zebras' in explorer.lookup('zebras')


def test_a_search_that_failed_discards_the_page_the_last_one_found(explorer, monkeypatch):
    """Lookup reads the last successful search, so a failed search has to clear it.

    Otherwise a Lookup after a failed Search silently reads the previous page, and
    the agent is answered about an article it did not ask for.
    """
    class Page:
        content = 'Telemundo is a television network mentioning zebras.'
        url = 'https://en.wikipedia.org/wiki/Telemundo'

    monkeypatch.setattr(wikipedia, 'page', lambda title, **kwargs: Page())
    explorer.search('Telemundo')
    assert 'zebras' in explorer.lookup('zebras'), 'the first search should have loaded'

    monkeypatch.setattr(wikipedia, 'page', lambda title, **kwargs: (_ for _ in ()).throw(
        wikipedia.PageError('no such page')))
    monkeypatch.setattr(wikipedia, 'search', lambda title: [])
    explorer.search('Nonexistent Title')

    with pytest.raises(ValueError, match='without a successful search'):
        explorer.lookup('zebras')


# ── the identity the endpoint answers to ──────────────────────────────────────

def test_the_package_default_user_agent_is_replaced():
    """Wikimedia rate limits that agent globally, so it is not usable.

    `wikipedia==1.4.0` sends `wikipedia (https://github.com/goldsmith/Wikipedia/)`,
    which every user of that unmaintained package shares. The endpoint answers it
    with 429 and a text/plain body, measured here as 0 of 3 searches reaching
    Wikipedia against 3 of 3 under an agent naming this project.
    """
    import tasks.envs.utils  # noqa: F401  (setting the agent is an import effect)

    agent = wikipedia.wikipedia.USER_AGENT

    assert 'goldsmith/Wikipedia' not in agent, 'the shared default agent is rate limited'
    assert 'Intrinsic-Memory' in agent, f'the agent should name this project, not {agent!r}'


def test_the_api_is_reached_over_https():
    import tasks.envs.utils  # noqa: F401

    assert wikipedia.wikipedia.API_URL.startswith('https://')


def test_a_failure_to_even_list_similar_titles_is_not_a_second_failure(
    explorer, monkeypatch
):
    """The similar titles are a courtesy. A page that is genuinely absent is
    still reported as absent when the suggestion call fails too."""
    monkeypatch.setattr(wikipedia, 'page', lambda title, **kwargs: (_ for _ in ()).throw(
        wikipedia.PageError('no such page')))
    monkeypatch.setattr(wikipedia, 'search', lambda title, **kwargs: (_ for _ in ()).throw(
        ValueError('rate limited')))

    result = explorer.search('Nonexistent Title')

    assert 'Could not find [Nonexistent Title]' in result


def test_a_transient_transport_failure_is_retried(explorer, monkeypatch):
    """One bad response must not lose the search.

    The API answers a burst of requests with a non-JSON body, which the
    `wikipedia` package raises as a bare JSONDecodeError. The window is seconds
    wide, so asking again inside the same search gets the page.
    """
    page, attempts = raising(
        ValueError('Expecting value: line 1 column 1 (char 0)'),
        ValueError('Expecting value: line 1 column 1 (char 0)'),
    )
    monkeypatch.setattr(wikipedia, 'page', page)

    result = explorer.search('Telemundo')

    assert result == 'the page text', f'gave up after {len(attempts)} attempts'
    assert len(attempts) == 3, f'made {len(attempts)} attempts'


def test_an_absent_page_is_not_retried(explorer, monkeypatch):
    """A page that does not exist is an answer, and asking again spends the rate
    limit to be told the same thing."""
    page, attempts = raising(wikipedia.PageError('no such page'))
    monkeypatch.setattr(wikipedia, 'page', page)
    monkeypatch.setattr(wikipedia, 'search', lambda title: [])

    explorer.search('Nonexistent Title')

    assert len(attempts) == 1, f'made {len(attempts)} attempts'


def test_the_retries_are_bounded(explorer, monkeypatch):
    """An endpoint that is down for good must not hold the episode open."""
    page, attempts = raising(*[
        ValueError('Expecting value: line 1 column 1 (char 0)')
        for _ in range(utils.WIKIPEDIA_ATTEMPTS + 1)
    ])
    monkeypatch.setattr(wikipedia, 'page', page)

    with pytest.raises(WikipediaUnavailable):
        explorer.search('Telemundo')

    assert len(attempts) == utils.WIKIPEDIA_ATTEMPTS, f'made {len(attempts)} attempts'


def test_the_package_fetches_through_the_status_aware_wrapper():
    """The wrapper is installed by importing the module, once however many times
    the repo is on the path under a different name."""
    assert hasattr(wikipedia.wikipedia.requests, 'delegate')
    assert not hasattr(wikipedia.wikipedia.requests.delegate, 'delegate'), 'wrapped twice'


def test_a_refusal_is_raised_rather_than_parsed_as_json(monkeypatch):
    """429 must not reach the package's `r.json()`.

    Wikimedia answers a burst with 429, an HTML body and a `Retry-After`. The
    package parses every response as JSON regardless, so the status and the wait
    it asked for were both lost - the caller saw a bare JSONDecodeError.
    """
    fetcher = wikipedia.wikipedia.requests
    monkeypatch.setattr(fetcher, 'delegate', SimpleNamespace(get=lambda *a, **k: Throttled()))

    with pytest.raises(RuntimeError) as throttled:
        fetcher.get('https://en.wikipedia.org/w/api.php')

    assert throttled.value.retry_after == 11


def test_a_refusal_that_names_no_wait_falls_back_to_the_backoff(monkeypatch):
    """`Retry-After` is optional, and may be an HTTP-date rather than seconds."""
    dated = Throttled()
    dated.headers = {'Retry-After': 'Wed, 21 Oct 2026 07:28:00 GMT'}
    fetcher = wikipedia.wikipedia.requests
    monkeypatch.setattr(fetcher, 'delegate', SimpleNamespace(get=lambda *a, **k: dated))

    with pytest.raises(RuntimeError) as throttled:
        fetcher.get('https://en.wikipedia.org/w/api.php')

    assert throttled.value.retry_after == utils.WIKIPEDIA_RETRY_SECONDS


def test_a_response_the_api_answered_is_passed_through(monkeypatch):
    """Only a refusal raises: an answer is the package's to parse."""
    answered = SimpleNamespace(status_code=200, headers={}, json=lambda: {'query': {}})
    fetcher = wikipedia.wikipedia.requests
    monkeypatch.setattr(fetcher, 'delegate', SimpleNamespace(get=lambda *a, **k: answered))

    assert fetcher.get('https://en.wikipedia.org/w/api.php') is answered


def test_a_throttled_search_waits_the_time_the_api_asked_for(explorer, monkeypatch, waits):
    """The wait is unguessable, so it is read rather than assumed.

    Wikimedia asked for eleven seconds where the blind backoff would have spent
    about three, retried inside the window and failed all four attempts.
    """
    page, attempts = raising(utils.WikipediaThrottled(11))
    monkeypatch.setattr(wikipedia, 'page', page)

    result = explorer.search('Telemundo')

    assert result == 'the page text', f'gave up after {len(attempts)} attempts'
    assert waits and waits[0] >= 11, f'waited {waits}'


def test_a_transport_failure_that_asks_for_nothing_backs_off(explorer, monkeypatch, waits):
    """A fault with no `Retry-After` still has to wait, or the retry is a no-op."""
    page, _ = raising(ValueError('Expecting value: line 1 column 1 (char 0)'))
    monkeypatch.setattr(wikipedia, 'page', page)

    explorer.search('Telemundo')

    assert waits and waits[0] > 0, f'waited {waits}'


def test_concurrent_workers_do_not_wait_in_lockstep(explorer, monkeypatch, waits):
    """Every worker of a sweep meets the same burst and is handed the same wait.

    Waiting exactly that long sends the whole sweep back at the API together, so
    the waits are jittered apart.
    """
    for _ in range(12):
        page, _ = raising(utils.WikipediaThrottled(11))
        monkeypatch.setattr(wikipedia, 'page', page)
        explorer.search('Telemundo')

    assert len(set(waits)) > 1, f'every worker waited {waits[0]}'


def test_the_title_asked_for_is_the_title_fetched(explorer, monkeypatch):
    """The package's `auto_suggest` default reports existing pages as absent.

    It replaces the requested title with Wikipedia's spelling suggestion and
    fetches that, so `L.A. Reid`, `Anne Rice`, `Nine Inch Nails`, `Damon Albarn`
    and `Brad Wilk` all raised PageError against the live API - and the agent was
    told the page did not exist while being handed its exact title as a similar
    one to try instead.
    """
    asked = {}

    class Page:
        content = 'the page text'
        url = 'https://en.wikipedia.org/wiki/L.A._Reid'

    def page(title, **kwargs):
        asked.update(title=title, **kwargs)
        return Page()

    monkeypatch.setattr(wikipedia, 'page', page)
    explorer.search('L.A. Reid')

    assert asked['auto_suggest'] is False, f'fetched with {asked}'
