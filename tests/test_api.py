import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_update_bound(client: TestClient) -> None:
    response = client.post(
        "/api/update-bound",
        json={"gamma": 0.1, "candidate_distance": 3.2},
    )
    assert response.status_code == 200
    assert response.json()["maximum_update_norm"] == 0.32000000000000006
    assert response.json()["bounded"] is True


def test_embedding_fusion(client: TestClient) -> None:
    response = client.post(
        "/api/fuse",
        json={
            "video": [[1.0, 0.0, 0.5, -0.5], [0.2, 0.1, 0.0, 0.3]],
            "query": [[0.4, 0.1, 0.8, -0.2]],
            "gamma": 0.1,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["gamma"] == 0.1
    assert payload["diagnostics"]["video_max_bound_violation"] <= 1e-6
    assert len(payload["fused_video"]) == 2
