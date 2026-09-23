from fastapi.testclient import TestClient

from xg_model.api.app import app

client = TestClient(app)


def post_shot(**details):
    return client.post("/api/xg", json={"position": {"x": 108, "y": 40}, **details})


def test_health():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_returns_a_probability():
    response = post_shot()

    assert response.status_code == 200
    assert 0 < response.json()["xg"] < 1


def test_header_is_worse_than_foot():
    assert post_shot(body_part="Head").json()["xg"] < post_shot().json()["xg"]


def test_defenders_are_taken_into_account():
    with_defender = post_shot(defenders=[{"x": 113, "y": 40}]).json()["xg"]

    assert with_defender < post_shot().json()["xg"]


def test_position_outside_the_pitch_is_rejected():
    response = client.post("/api/xg", json={"position": {"x": 200, "y": 40}})

    assert response.status_code == 422


def test_unknown_body_part_is_rejected():
    assert post_shot(body_part="Knee").status_code == 422


def test_too_many_defenders_are_rejected():
    defenders = [{"x": 110, "y": 40}] * 11

    assert post_shot(defenders=defenders).status_code == 422
