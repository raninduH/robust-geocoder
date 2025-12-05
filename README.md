
# Robust Geocoder


A comprehensive geocoding library with multiple fallback providers. Supports single location queries and parallel batch processing.

## Features
- Multiple geocoding providers with fallback logic:
  1. Geopy Nominatim (OpenStreetMap)
  2. Wikidata SPARQL
  3. Photon API
  4. DuckDuckGo Instant Answer (free)
  5. Serper API (optional, requires API key)
- Batch geocoding with parallel processing
- Input cleaning and coordinate validation
- Provider usage statistics
- Separate function for postal code geocoding

## Installation
```bash
pip install .
```

## Usage
```python
from robust_geocoder import GeocodeFallback

# Initialize for Sri Lanka
geocoder = GeocodeFallback(country_code='LK', bounds=(5.9, 9.9, 79.5, 82.0))

# Single location (city/district)
result = geocoder.geocode_location("Colombo", verbose=True)

# Batch geocoding
locations = [
  {'location': 'Colombo', 'district': 'Colombo'},
  {'location': 'Kandy', 'district': 'Kandy'},
]
results = geocoder.geocode_batch(locations, num_threads=4, verbose=True)

# Postal code lookup (returns coordinates for valid postal codes)
postal_result = geocoder.geocode_postal_code("00100")
print(postal_result)
```

## Requirements
- pandas
- pgeocode
- requests
- python-dotenv
- geopy
- SPARQLWrapper (optional, for Wikidata)
- duckduckgo-search (optional, for DuckDuckGo)

## Environment Variables
- `SERPER_API_KEY` (optional, for Serper API)

## License
MIT
