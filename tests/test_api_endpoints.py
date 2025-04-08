
import unittest
from app import app
import json

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_calculate_price(self):
        response = self.app.get('/api/v1/calculate_price', query_string={
            'commodity': 'Wheat',
            'region': 'North India',
            'quality_params': json.dumps({'moisture': 12.0})
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('final_price', data)
        self.assertIn('base_price', data)
    
    def test_price_history(self):
        response = self.app.get('/api/v1/price_history', query_string={
            'commodity': 'Wheat',
            'region': 'North India',
            'days': 30
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
    
    def test_wizx_index(self):
        response = self.app.get('/api/v1/wizx_index', query_string={
            'commodity': 'Wheat'
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('index_value', data)

if __name__ == '__main__':
    unittest.main()
