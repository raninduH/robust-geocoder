import unittest
from geocode_fallback_library import GeocodeFallback

class TestPhoton(unittest.TestCase):
    def setUp(self):
        self.geocoder = GeocodeFallback(country_code='LK', bounds=(5.9, 9.9, 79.5, 82.0))

    def test_photon(self):
        result = self.geocoder.geocode_with_photon("Colombo")
        self.assertIn(result['success'], [True, False])
        if result['success']:
            self.assertIsNotNone(result['latitude'])
            self.assertIsNotNone(result['longitude'])

if __name__ == "__main__":
    unittest.main()
