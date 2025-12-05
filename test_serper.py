import unittest
import os
from geocode_fallback_library import GeocodeFallback

class TestSerper(unittest.TestCase):
    def setUp(self):
        self.geocoder = GeocodeFallback(country_code='LK', bounds=(5.9, 9.9, 79.5, 82.0))

    def test_serper(self):
        # Only run if SERPER_API_KEY is set
        if not os.getenv('SERPER_API_KEY'):
            self.skipTest("SERPER_API_KEY not set in environment")
        result = self.geocoder.geocode_with_serper("Colombo")
        self.assertIn(result['success'], [True, False])
        if result['success']:
            self.assertIsNotNone(result['latitude'])
            self.assertIsNotNone(result['longitude'])

if __name__ == "__main__":
    unittest.main()
