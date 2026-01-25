import pytest
from app import app, predict_fish, calculate_survival_probability
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert 'Fish Recognition' in rv.json['message']

def test_species_info():
    from app import SPECIES_INFO
    assert 'devario_malabaricus' in SPECIES_INFO

def test_survival_calculation():
    result = calculate_survival_probability('devario_malabaricus', {
        'pH': 7.2, 'temperature_C': 25, 'dissolved_oxygen_mgL': 7.0
    })
    assert result['probability'] > 80