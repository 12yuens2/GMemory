"""A failed Wikipedia search is reported, not passed off as a page that is absent.

The distinction decides whether a run's results mean anything. A search that
cannot reach Wikipedia and one for a page that does not exist both leave the
agent with nothing, so if the two are reported alike a run against a blocked
endpoint scores every episode zero and reads as a weak agent rather than a
broken experiment.
"""

import wikipedia

import pytest

from tasks.envs.utils import LangChainWiki, WikipediaUnavailable


@pytest.fixture
def explorer(monkeypatch):
    return LangChainWiki()


def test_a_page_that_does_not_exist_is_reported_as_absent(explorer, monkeypatch):
    monkeypatch.setattr(wikipedia, 'page', lambda title: (_ for _ in ()).throw(
        wikipedia.PageError('no such page')))
    monkeypatch.setattr(wikipedia, 'search', lambda title: ['Something else'])

    result = explorer.search('Nonexistent Title')

    assert 'Could not find [Nonexistent Title]' in result
    assert 'Something else' in result, 'the similar titles are what the agent searches next'


def test_a_disambiguation_is_reported_as_absent_with_the_alternatives(explorer, monkeypatch):
    monkeypatch.setattr(wikipedia, 'page', lambda title: (_ for _ in ()).throw(
        wikipedia.DisambiguationError('ambiguous', [])))
    monkeypatch.setattr(wikipedia, 'search', lambda title: ['Mercury (planet)'])

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
    monkeypatch.setattr(wikipedia, 'page', lambda title: (_ for _ in ()).throw(
        ValueError('Expecting value: line 1 column 1 (char 0)')))

    with pytest.raises(WikipediaUnavailable, match='could not be reached'):
        explorer.search('Telemundo')


def test_the_transport_failure_names_what_went_wrong(explorer, monkeypatch):
    """The message is the only place the cause survives, so it carries it."""
    monkeypatch.setattr(wikipedia, 'page', lambda title: (_ for _ in ()).throw(
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

    def page(title):
        calls.append(title)
        return Page()

    monkeypatch.setattr(wikipedia, 'page', page)
    explorer.search('Telemundo')

    assert calls == ['Telemundo'], f'fetched {len(calls)} times'


def test_a_search_that_succeeds_keeps_the_page_for_lookup(explorer, monkeypatch):
    class Page:
        content = 'First paragraph.\n\nSecond paragraph mentioning zebras.'
        url = 'https://en.wikipedia.org/wiki/Telemundo'

    monkeypatch.setattr(wikipedia, 'page', lambda title: Page())
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

    monkeypatch.setattr(wikipedia, 'page', lambda title: Page())
    explorer.search('Telemundo')
    assert 'zebras' in explorer.lookup('zebras'), 'the first search should have loaded'

    monkeypatch.setattr(wikipedia, 'page', lambda title: (_ for _ in ()).throw(
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
    monkeypatch.setattr(wikipedia, 'page', lambda title: (_ for _ in ()).throw(
        wikipedia.PageError('no such page')))
    monkeypatch.setattr(wikipedia, 'search', lambda title: (_ for _ in ()).throw(
        ValueError('rate limited')))

    result = explorer.search('Nonexistent Title')

    assert 'Could not find [Nonexistent Title]' in result
