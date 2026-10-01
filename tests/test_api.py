"""
Automated Integration and Validation Tests for FastAPI Service.

Tests:
1. Root '/' endpoint metadata retrieval.
2. Health check '/health' endpoint.
3. Valid inference prediction via POST '/predict'.
4. Boundary/Range validation: Negative SOC (-20).
5. Boundary/Range validation: Excessive SOC (>100).
6. Missing required fields.
7. Invalid categorical entries.

Author: Academic Machine Learning Mentorship & Research Team
Project: EV Charging Demand Prediction Using Ensemble Machine Learning
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

# Ensure root and src are on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for path in [PROJECT_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root endpoint returns project metadata and status 200."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert data["target_variable"] == "Energy Consumed (kWh)"
    assert "best_model_name" in data


def test_health_endpoint():
    """Verify health endpoint indicates healthy status and model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_valid_prediction():
    """Verify inference returns valid kWh prediction for standard EV arrival payload."""
    payload = {
        "battery_capacity": 60.0,
        "soc_start": 25.0,
        "vehicle_age": 3.0,
        "temperature": 28.0,
        "distance_driven": 120.0,
        "charger_type": "DC Fast Charger",
        "time_of_day": "Evening",
        "day_of_week": "Saturday",
        "user_type": "Commuter",
        "vehicle_model": "Tesla Model 3",
        "location": "Chicago"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_energy_kwh" in data
    assert isinstance(data["predicted_energy_kwh"], float)
    assert data["predicted_energy_kwh"] >= 0.0
    assert data["unit"] == "kWh"
    assert data["status"] == "success"


def test_invalid_negative_soc():
    """Verify API rejects physically invalid negative SOC with HTTP 422."""
    payload = {
        "battery_capacity": 60.0,
        "soc_start": -20.0,  # Invalid
        "vehicle_age": 3.0,
        "temperature": 28.0,
        "distance_driven": 120.0,
        "charger_type": "DC Fast Charger",
        "time_of_day": "Evening",
        "day_of_week": "Saturday",
        "user_type": "Commuter"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_invalid_excessive_soc():
    """Verify API rejects SOC > 100 with HTTP 422."""
    payload = {
        "battery_capacity": 60.0,
        "soc_start": 150.0,  # Invalid
        "vehicle_age": 3.0,
        "temperature": 28.0,
        "distance_driven": 120.0,
        "charger_type": "DC Fast Charger",
        "time_of_day": "Evening",
        "day_of_week": "Saturday",
        "user_type": "Commuter"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_required_field():
    """Verify API rejects payload missing mandatory fields with HTTP 422."""
    payload = {
        # 'battery_capacity' is missing
        "soc_start": 40.0,
        "vehicle_age": 2.0,
        "temperature": 20.0,
        "distance_driven": 80.0,
        "charger_type": "Level 2",
        "time_of_day": "Morning",
        "day_of_week": "Monday",
        "user_type": "Commuter"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_categorical_value():
    """Verify API rejects invalid categorical enum values with HTTP 422."""
    payload = {
        "battery_capacity": 75.0,
        "soc_start": 50.0,
        "vehicle_age": 1.0,
        "temperature": 22.0,
        "distance_driven": 50.0,
        "charger_type": "SolarSuperLaser",  # Invalid categorical
        "time_of_day": "Morning",
        "day_of_week": "Wednesday",
        "user_type": "Commuter"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
