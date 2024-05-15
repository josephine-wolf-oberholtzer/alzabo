from uuid import uuid4

import pytest


@pytest.fixture
def job_id():
    return str(uuid4())


@pytest.fixture
def staging_id():
    return str(uuid4())
