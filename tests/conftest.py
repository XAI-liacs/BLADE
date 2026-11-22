import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_env_cache(tmp_path_factory):
    """Keep cached evaluation environments of the test run out of ~/.cache.

    One cache is shared by the whole session, so tests still reuse each
    other's environments.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("IOHBLADE_CACHE_DIR", str(tmp_path_factory.mktemp("iohblade-cache")))
        yield
