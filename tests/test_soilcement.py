import numpy as np
from soilcement import load_corpus, equivalent_cement, features, HierarchicalLaw
from soilcement.cli import main


def test_corpus():
    df = load_corpus()
    assert len(df) == 1372 and df.deposit.nunique() == 38


def test_equivalent_cement():
    assert np.allclose(equivalent_cement([10, 10, 10], ["opc", "slag", "opc"], [0, 0, 20]), [10, 17.2, 11.5])


def test_law_updates_towards_trials():
    law = HierarchicalLaw()
    Xq = features([150], [20], [28], 100, 60)
    m0, lo0, hi0 = law.predict(np.zeros((0, 6)), np.zeros(0), Xq, 100)
    Xt = features([150, 150, 150], [10, 20, 30], [28, 28, 28], 100, 60)
    yt = np.log([300.0, 600.0, 900.0])
    m3, lo3, hi3 = law.predict(Xt, yt, Xq, 100)
    assert abs(np.exp(m3[0]) - 600) < abs(np.exp(m0[0]) - 600) + 1e-6   # moves towards the evidence
    assert (hi3 - lo3)[0] < (hi0 - lo0)[0]                               # interval narrows


def test_cli_law(tmp_path, capsys):
    q = tmp_path / "q.csv"; q.write_text("water_content_pct,binder_pct,curing_days\n150,20,28\n")
    main(["predict", "--ll", "100", "--pi", "60", "--query", str(q), "--model", "law"])
    assert "qu_pred_kPa" in capsys.readouterr().out
