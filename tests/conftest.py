import pytest
from tcc.ingest import connect, init_db
from tcc.pseudonymize import Pseudonymizer
from tcc.synthetic import generate, ATENDENTES

SECRET = "segredo-de-teste-123456"


@pytest.fixture
def pseud():
    return Pseudonymizer(secret=SECRET, attendant_names=ATENDENTES)


@pytest.fixture
def db(tmp_path):
    conn = connect(str(tmp_path / "t.db"))
    init_db(conn)
    return conn


@pytest.fixture(scope="session")
def syn_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    generate(str(d), n_leads=80, seed=11, secret=SECRET)
    return d
