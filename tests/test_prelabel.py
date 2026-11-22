import csv
import json

from tcc.experiments.llm import Cache, Provider
from tcc.ingest import ingest_file
from tcc.labels import import_labels
from tcc.prelabel import _demora, run, suggest, to_row_values


class FakeProvider(Provider):
    name, model = "fake", "fake-v1"

    def __init__(self, answer):
        self.answer = answer; self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return self.answer


GOOD = json.dumps({"escopo": "em_escopo", "funil": "intencao", "objecao": ["preco"], "gargalo": []})


def test_demora_regra_do_guia():
    assert _demora("", "") is True            # atendente nunca respondeu
    assert _demora(5.0, 10.0) is False
    assert _demora(5.0, 31.0) is True         # qualquer tempo > 30


def test_demora_vem_da_regra_nao_do_modelo():
    v = to_row_values({"escopo": "em_escopo", "funil": "intencao", "objecao": [], "gargalo": []}, 45.0, 10.0)
    assert v["gargalo"] == "demora_resposta" and v["objecao"] == "nenhuma"
    # mesmo que o modelo tente marcar demora_resposta, a regra manda (sem tempo alto: não marca)
    v2 = to_row_values({"escopo": "em_escopo", "funil": "informativo", "objecao": [],
                        "gargalo": ["demora_resposta"]}, 5.0, 5.0)
    assert v2["gargalo"] == "nenhuma"


def test_fora_de_escopo_nao_tem_funil():
    v = to_row_values({"escopo": "fora_de_escopo", "funil": "agendado", "objecao": [], "gargalo": []}, 1.0, 1.0)
    assert v["funil"] == ""


def test_prelabel_aceita_novos_rotulos_v1_1():
    v = to_row_values({"escopo": "em_escopo", "funil": "informativo", "objecao": ["servico_nao_oferecido"],
                       "gargalo": ["sem_vaga"], "para_terceiro": "sim"}, 1.0, 1.0)
    assert v["objecao"] == "servico_nao_oferecido" and v["gargalo"] == "sem_vaga" and v["para_terceiro"] == "sim"


def test_sugestao_invalida_vira_none(tmp_path):
    cache = Cache(str(tmp_path / "c.sqlite"))
    assert suggest(FakeProvider("nao e json"), cache, "texto", 1.0, 1.0, retries=1) is None
    bad = json.dumps({"escopo": "talvez", "qualificacao": None, "objecao": [], "gargalo": []})
    assert suggest(FakeProvider(bad), cache, "texto", 1.0, 1.0, retries=0) is None


def _db_com_conversas(db, pseud, tmp_path, n=8):
    for i in range(n):
        f = tmp_path / f"Conversa do WhatsApp com Pessoa {i}.txt"
        f.write_text(f"[18/08/2026 09:3{i % 10}:00] Pessoa {i}: Quanto custa a sessao numero {i}?\n"
                     f"[18/08/2026 09:4{i % 10}:00] Clinica Vida Plena: R$ 200,00\n", encoding="utf-8")
        ingest_file(db, str(f), pseud)


def test_run_separa_cegas_e_a2_nunca_ve_sugestao(db, pseud, tmp_path):
    _db_com_conversas(db, pseud, tmp_path)
    prov = FakeProvider(GOOD)
    a1, a2 = str(tmp_path / "a1.csv"), str(tmp_path / "a2.csv")
    st = run(db, prov, Cache(str(tmp_path / "c.sqlite")), a1, a2, blind_frac=0.25, seed=5)
    r1 = list(csv.DictReader(open(a1, encoding="utf-8-sig")))
    r2 = list(csv.DictReader(open(a2, encoding="utf-8-sig")))
    assert st["total"] == len(r1) and st["cegas"] == len(r2) > 0
    assert all(r["escopo"] == "" and r["pre_rotulo"] == "cega" for r in r2)      # a2: sempre em branco
    cegas_a1 = [r for r in r1 if r["pre_rotulo"] == "cega"]
    assert all(r["escopo"] == "" for r in cegas_a1)                              # cegas em branco também no a1
    assert {r["conversation_id"] for r in r2} == {r["conversation_id"] for r in cegas_a1}
    assert prov.calls == st["sugeridas"]                                          # modelo não vê as cegas
    assert st["sugeridas"] + st["cegas"] + st["invalidas"] == st["total"]


def test_csv_a1_e_importavel(db, pseud, tmp_path):
    _db_com_conversas(db, pseud, tmp_path)
    a1, a2 = str(tmp_path / "a1.csv"), str(tmp_path / "a2.csv")
    run(db, FakeProvider(GOOD), Cache(str(tmp_path / "c.sqlite")), a1, a2, blind_frac=0.25, seed=5)
    res = import_labels(db, a1, "a1")
    assert res["errors"] == [] and res["imported"] > 0
