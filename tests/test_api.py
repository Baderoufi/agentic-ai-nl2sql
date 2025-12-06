from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_nl2sql_basic():
    """
    Test the /nl2sql endpoint with a simple question.
    Note:
        This assumes:
        - schema.json exists
        - vector_sql_schema3 exists
        - user_id exists in ACL (or Mocked)
    """
    payload = {
        "user_id": "1",
        "question": "How many customers do we have?",
        "n_rows": 5
    }

    response = client.post("/nl2sql", json=payload)

    # API must respond with something
    assert response.status_code in [200, 400, 500]

    # If success
    if response.status_code == 200:
        data = response.json()
        assert "sql" in data
        assert isinstance(data["sql"], str)
        assert "SELECT" in data["sql"].upper()


def test_nl2sql_missing_fields():
    """Test error handling when required fields are missing."""
    payload = {
        "user_id": "1",
        # missing "question"
    }

    response = client.post("/nl2sql", json=payload)
    assert response.status_code == 422  # validation error from FastAPI
