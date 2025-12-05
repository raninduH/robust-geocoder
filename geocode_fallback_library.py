"""
Robust Geocoder
A comprehensive geocoding library with multiple fallback providers.
Supports single location queries and parallel batch processing.

Hierarchy:
1. pgeocode (postal codes)
2. Geopy Nominatim (OpenStreetMap)
3. Wikidata SPARQL
4. Photon API
5. DuckDuckGo Instant Answer (free)
6. Serper API (optional, requires API key)
"""

import pandas as pd
import pgeocode
import re
import requests
import os
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from SPARQLWrapper import SPARQLWrapper, JSON
    WIKIDATA_AVAILABLE = True
except ImportError:
    WIKIDATA_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False

# Load environment variables
load_dotenv()

# Global configuration
PHOTON_URL = "https://photon.komoot.io/api/"
SERPER_URL = "https://google.serper.dev/search"
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

# Thread-safe lock for rate limiting
rate_limit_lock = threading.Lock()
last_request_time = {'geopy': 0, 'wikidata': 0}


class GeocodeFallback:
    """
    A robust geocoding class with multiple provider fallbacks.
    """
    
    def __init__(self, country_code='LK', bounds=(5.9, 9.9, 79.5, 82.0), user_agent="robust_geocoder"):
        """
        Initialize the geocoder with country-specific settings.
        
        Args:
            country_code: ISO country code for pgeocode (default: 'LK' for Sri Lanka)
            bounds: Tuple of (min_lat, max_lat, min_lon, max_lon) for coordinate validation
            user_agent: User agent string for API requests
        """
        self.country_code = country_code
        self.min_lat, self.max_lat, self.min_lon, self.max_lon = bounds
        self.user_agent = user_agent
        
        # Initialize geocoders
        self.nomi = pgeocode.Nominatim(country_code)
        self.geolocator = Nominatim(user_agent=user_agent)
        
        # Statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'pgeocode': 0,
            'geopy': 0,
            'wikidata': 0,
            'photon': 0,
            'duckduckgo': 0,
            'serper': 0,
            'failed': 0
        }
    
    def clean_location(self, location):
        """Remove special characters and clean location string, but preserve spaces and numbers."""
        cleaned = re.sub(r'\(.*?\)', '', str(location))
        cleaned = re.sub(r'[()]', '', cleaned)
        # Only remove special characters except letters, numbers, comma, space, period, hyphen
        cleaned = re.sub(r'[^a-zA-Z0-9, .-]', '', cleaned)
        return cleaned.strip()
    
    def remove_numbers(self, location):
        """Remove all numbers from location string, keeping only words and spaces."""
        cleaned = re.sub(r'\d+', '', location)
        return cleaned.strip()
    
    def validate_coordinates(self, lat, lon):
        """Check if coordinates are within country bounds."""
        return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon
    

    def geocode_postal_code(self, postal_code):
        """Geocode using pgeocode for postal codes only."""
        try:
            result = self.nomi.query_postal_code(postal_code)
            lat = result['latitude']
            lon = result['longitude']
            if pd.notnull(lat) and pd.notnull(lon) and self.validate_coordinates(lat, lon):
                return {
                    'latitude': lat,
                    'longitude': lon,
                    'success': True,
                    'source': 'postal-code',
                    'confidence': 'high'
                }
        except Exception:
            pass
        return {'success': False, 'source': 'postal-code', 'error': 'No result'}
    
    def geocode_with_geopy(self, location, district=None, country=None, region=None):
        """Geocode using Geopy Nominatim (OpenStreetMap)."""
        try:
            # Rate limiting
            with rate_limit_lock:
                time_since_last = time.time() - last_request_time['geopy']
                if time_since_last < 1:
                    time.sleep(1 - time_since_last)
                last_request_time['geopy'] = time.time()
            # Build query
            query_parts = [location]
            if region:
                query_parts.append(region)
            elif district:
                query_parts.append(district)
            if country:
                query_parts.append(country)
            query = ", ".join(query_parts)
            result = self.geolocator.geocode(query, timeout=10)
            if result:
                lat, lon = result.latitude, result.longitude
                if self.validate_coordinates(lat, lon):
                    return {
                        'latitude': lat,
                        'longitude': lon,
                        'success': True,
                        'source': 'geopy-nominatim',
                        'confidence': 'high',
                        'address': result.address
                    }
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            pass
        except Exception as e:
            pass
        return {'success': False, 'source': 'geopy-nominatim', 'error': 'No result'}
    
    def geocode_with_wikidata(self, location, country_qid='Q854', region=None):
        """Geocode using Wikidata SPARQL query."""
        if not WIKIDATA_AVAILABLE:
            return {'success': False, 'source': 'wikidata', 'error': 'SPARQLWrapper not installed'}
        
        try:
            # Rate limiting
            with rate_limit_lock:
                time_since_last = time.time() - last_request_time['wikidata']
                if time_since_last < 0.5:
                    time.sleep(0.5 - time_since_last)
                last_request_time['wikidata'] = time.time()
            sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
            sparql.setReturnFormat(JSON)
            label = f"{location}, {region}" if region else location
            query = f"""
            SELECT ?place ?placeLabel ?coord ?lat ?lon WHERE {{
              ?place rdfs:label \"{label}\"@en .
              ?place wdt:P17 wd:{country_qid} .
              ?place wdt:P625 ?coord .
              ?place p:P625 ?coordinate .
              ?coordinate psv:P625 ?coordinate_node .
              ?coordinate_node wikibase:geoLatitude ?lat .
              ?coordinate_node wikibase:geoLongitude ?lon .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"en\". }}
            }} LIMIT 1
            """
            sparql.setQuery(query)
            results = sparql.query().convert()
            if results.get('results', {}).get('bindings'):
                binding = results['results']['bindings'][0]
                lat = float(binding['lat']['value'])
                lon = float(binding['lon']['value'])
                if self.validate_coordinates(lat, lon):
                    return {
                        'latitude': lat,
                        'longitude': lon,
                        'success': True,
                        'source': 'wikidata',
                        'confidence': 'high',
                        'label': binding.get('placeLabel', {}).get('value', label)
                    }
        except Exception as e:
            pass
        return {'success': False, 'source': 'wikidata', 'error': 'No result'}
    
    def geocode_with_photon(self, location, district=None, country=None, region=None):
        """Geocode using Photon API (OpenStreetMap)."""
        try:
            query_parts = [location]
            if region:
                query_parts.append(region)
            elif district:
                query_parts.append(district)
            if country:
                query_parts.append(country)
            query = ", ".join(query_parts)
            params = {'q': query, 'limit': 1}
            response = requests.get(PHOTON_URL, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'features' in data and len(data['features']) > 0:
                    feature = data['features'][0]
                    coords = feature['geometry']['coordinates']
                    lat, lon = coords[1], coords[0]
                    if self.validate_coordinates(lat, lon):
                        return {
                            'latitude': lat,
                            'longitude': lon,
                            'success': True,
                            'source': 'photon',
                            'confidence': 'medium'
                        }
        except Exception as e:
            pass
        return {'success': False, 'source': 'photon', 'error': 'No result'}
    
    def extract_coordinates_from_text(self, text):
        """Extract latitude and longitude from text using multiple regex patterns."""
        patterns = [
            # Pattern 1: "6.4647° N, 80.6160° E"
            r'([0-9]+\.[0-9]+)°?\s*[Nn][,\s]+([0-9]+\.[0-9]+)°?\s*[Ee]',
            # Pattern 2: "8.5873638 latitude; 81.2152121 longitude"
            r'([0-9]+\.[0-9]+)\s*latitude[;\s,]+([0-9]+\.[0-9]+)\s*longitude',
            # Pattern 3: 'latitude": 6.4647, "longitude": 80.6160'
            r'["]?latitude["]?[:\s]+([0-9]+\.[0-9]+).*?["]?longitude["]?[:\s]+([0-9]+\.[0-9]+)',
            # Pattern 4: "Lat: 6.4647, Lon: 80.6160" or "Lat=6.4647, Lon=80.6160"
            r'[Ll]at[:=\s]+([0-9]+\.[0-9]+)[,;\s]+[Ll]on[g]?(?:itude)?[:=\s]+([0-9]+\.[0-9]+)',
            # Pattern 5: "6.4647, 80.6160" (comma separated, both in valid range)
            r'([0-9]+\.[0-9]+)[,\s]+([0-9]+\.[0-9]+)',
            # Pattern 6: "N 6.4647 E 80.6160"
            r'[Nn]\s*([0-9]+\.[0-9]+)[,;\s]+[Ee]\s*([0-9]+\.[0-9]+)',
            # Pattern 7: "Latitude 6.4647 Longitude 80.6160"
            r'[Ll]atitude[:=\s]*([0-9]+\.[0-9]+)[,;\s]+[Ll]ongitude[:=\s]*([0-9]+\.[0-9]+)',
        ]
        text_str = str(text)
        for idx, pattern in enumerate(patterns):
            match = re.search(pattern, text_str, re.IGNORECASE)
            if match:
                try:
                    lat = float(match.group(1))
                    lon = float(match.group(2))
                    if self.validate_coordinates(lat, lon):
                        return (lat, lon)
                except Exception:
                    continue
        return None
    
    def geocode_with_duckduckgo(self, location, district=None, country=None, region=None):
        """Geocode using DuckDuckGo Instant Answer API (free)."""
        if not DUCKDUCKGO_AVAILABLE:
            return {'success': False, 'source': 'duckduckgo', 'error': 'duckduckgo-search not installed'}
        
        try:
            query_parts = [location]
            if region:
                query_parts.append(region)
            elif district:
                query_parts.append(district)
            if country:
                query_parts.append(country)
            query = " ".join(query_parts) + " latitude longitude"
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                for result in results:
                    # Try to extract from body text
                    text = result.get('body', '') + ' ' + result.get('title', '')
                    coords = self.extract_coordinates_from_text(text)
                    if coords:
                        return {
                            'latitude': coords[0],
                            'longitude': coords[1],
                            'success': True,
                            'source': 'duckduckgo',
                            'confidence': 'medium'
                        }
        except Exception as e:
            pass
        return {'success': False, 'source': 'duckduckgo', 'error': 'No result'}
    
    def geocode_with_serper(self, location, district=None, region=None):
        """Geocode using Serper API (requires API key)."""
        if not SERPER_API_KEY:
            return {'success': False, 'source': 'serper', 'error': 'No API key'}
        try:
            query_parts = [location]
            if region:
                query_parts.append(region)
            elif district:
                query_parts.append(district)
            query = ", ".join(query_parts) + " latitude and longitude"
            headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
            payload = {'q': query, 'gl': self.country_code.lower(), 'hl': 'en'}
            response = requests.post(SERPER_URL, headers=headers, json=payload, timeout=10)
            if response.status_code != 200:
                return {'success': False, 'source': 'serper', 'error': f'Status {response.status_code}'}
            data = response.json()
            # Try knowledge graph
            if 'knowledgeGraph' in data and 'coordinates' in data['knowledgeGraph']:
                kg = data['knowledgeGraph']
                lat = kg['coordinates'].get('latitude')
                lon = kg['coordinates'].get('longitude')
                if lat and lon and self.validate_coordinates(lat, lon):
                    return {
                        'latitude': lat,
                        'longitude': lon,
                        'success': True,
                        'source': 'serper-kg',
                        'confidence': 'high'
                    }
            # Try organic results
            if 'organic' in data:
                for result in data['organic'][:3]:
                    snippet = result.get('snippet', '')
                    coords = self.extract_coordinates_from_text(snippet)
                    if coords:
                        return {
                            'latitude': coords[0],
                            'longitude': coords[1],
                            'success': True,
                            'source': 'serper-organic',
                            'confidence': 'medium'
                        }
        except Exception as e:
            pass
        return {'success': False, 'source': 'serper', 'error': 'No result'}
    
    def geocode_location(self, location, district=None, country=None, region=None, verbose=False):
        """
        Geocode a single location using fallback hierarchy.
        
        Args:
            location: Location name to geocode
            district: Optional district/region name
            country: Optional country name
            verbose: Print detailed progress
        
        Returns:
            Dictionary with latitude, longitude, success status, and source
        """
        self.stats['total'] += 1

        if verbose:
            print(f"\n🔍 Geocoding: {location}")
            if district:
                print(f"   District: {district}")

        # Clean location (preserve numbers and spaces)
        clean_loc = self.clean_location(location)
        # If location is all numbers, treat as postal code
        if clean_loc.isdigit():
            result = self.geocode_postal_code(clean_loc)
            result['original_location'] = location
            result['cleaned_location'] = clean_loc
            result['district'] = district
            return result

        has_numbers = bool(re.search(r'\d', clean_loc))

        provider_name_final = None
        result_final = None

        def try_all_providers(loc_string):
            providers = [
                ('geopy', lambda: self.geocode_with_geopy(loc_string, district, country, region=region)),
                ('wikidata', lambda: self.geocode_with_wikidata(loc_string, region=region)),
                ('photon', lambda: self.geocode_with_photon(loc_string, district, country, region=region)),
                ('duckduckgo', lambda: self.geocode_with_duckduckgo(loc_string, district, country, region=region)),
            ]
            if SERPER_API_KEY:
                providers.append(('serper', lambda: self.geocode_with_serper(loc_string, district, region=region)))
            for provider_name, provider_func in providers:
                if verbose:
                    print(f"   → Trying {provider_name}...", end=" ")
                result = provider_func()
                if result['success']:
                    if verbose:
                        print(f"✓ Success! ({result['latitude']:.6f}, {result['longitude']:.6f})")
                    return result, provider_name
                if verbose:
                    print("✗ No result")
            return None, None

        # Try with numbers
        result, provider_name = try_all_providers(clean_loc)
        if result:
            provider_name_final = provider_name
            result_final = result
            cleaned_location_final = clean_loc
        elif has_numbers:
            clean_loc_no_numbers = self.remove_numbers(clean_loc)
            if clean_loc_no_numbers and clean_loc_no_numbers != clean_loc:
                if verbose:
                    print(f"   🔄 Retrying without numbers: {clean_loc_no_numbers}")
                result, provider_name = try_all_providers(clean_loc_no_numbers)
                if result:
                    provider_name_final = provider_name
                    result_final = result
                    cleaned_location_final = clean_loc_no_numbers

        if result_final:
            self.stats['success'] += 1
            if provider_name_final:
                self.stats[provider_name_final] += 1
            result_final['original_location'] = location
            result_final['cleaned_location'] = cleaned_location_final
            result_final['district'] = district
            if cleaned_location_final != clean_loc:
                result_final['note'] = 'Succeeded after removing numbers'
            return result_final

        self.stats['failed'] += 1
        if verbose:
            print(f"   ✗ All providers failed for: {location}")
        return {
            'success': False,
            'latitude': None,
            'longitude': None,
            'source': None,
            'original_location': location,
            'cleaned_location': clean_loc,
            'district': district,
            'error': 'All providers failed'
        }
    
    def geocode_batch(self, locations, num_threads=4, verbose=False):
        """
        Geocode multiple locations in parallel using thread pool.
        
        Args:
            locations: List of dictionaries with 'location', optional 'district' and 'country'
                      OR list of strings (location names only)
            num_threads: Number of parallel threads (default: 4)
            verbose: Print progress updates
        
        Returns:
            List of geocoding results
        """
        # Normalize input format
        if locations and isinstance(locations[0], str):
            locations = [{'location': loc} for loc in locations]
        
        total = len(locations)
        results = []
        completed = 0
        batch_success = 0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🚀 Starting batch geocoding")
            print(f"   Total locations: {total}")
            print(f"   Threads: {num_threads}")
            print(f"   Serper API: {'✓ Enabled' if SERPER_API_KEY else '✗ Disabled (no API key)'}")
            print(f"{'='*60}")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            future_to_loc = {
                executor.submit(
                    self.geocode_location,
                    loc.get('location'),
                    loc.get('district'),
                    loc.get('country'),
                    False  # Don't print in threads
                ): loc for loc in locations
            }
            
            # Process completed tasks
            for future in as_completed(future_to_loc):
                loc = future_to_loc[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if result['success']:
                        batch_success += 1
                    
                    if verbose and completed % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total - completed) / rate if rate > 0 else 0
                        print(f"📊 Progress: {completed}/{total} ({completed/total*100:.1f}%) | "
                              f"Success: {batch_success} | "
                              f"Rate: {rate:.1f}/s | "
                              f"ETA: {eta:.0f}s")
                
                except Exception as e:
                    results.append({
                        'success': False,
                        'latitude': None,
                        'longitude': None,
                        'source': None,
                        'original_location': loc.get('location'),
                        'error': str(e)
                    })
                    completed += 1
        
        elapsed = time.time() - start_time
        
        # Calculate batch statistics
        batch_stats = {
            'pgeocode': 0,
            'geopy': 0,
            'wikidata': 0,
            'photon': 0,
            'duckduckgo': 0,
            'serper': 0,
            'failed': 0
        }
        
        for res in results:
            if res['success']:
                source = res.get('source', '')
                if source == 'pgeocode':
                    batch_stats['pgeocode'] += 1
                elif 'geopy' in source:
                    batch_stats['geopy'] += 1
                elif source == 'wikidata':
                    batch_stats['wikidata'] += 1
                elif source == 'photon':
                    batch_stats['photon'] += 1
                elif source == 'duckduckgo':
                    batch_stats['duckduckgo'] += 1
                elif 'serper' in source:
                    batch_stats['serper'] += 1
            else:
                batch_stats['failed'] += 1
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ Batch geocoding complete!")
            print(f"   Total time: {elapsed:.1f}s")
            print(f"   Average: {elapsed/total:.2f}s per location")
            print(f"   Success rate: {batch_success/total*100:.1f}%")
            print(f"\n📈 Results by provider:")
            for provider in ['pgeocode', 'geopy', 'wikidata', 'photon', 'duckduckgo', 'serper']:
                count = batch_stats[provider]
                if count > 0:
                    print(f"   {provider:15s}: {count:4d} ({count/total*100:.1f}%)")
            print(f"   {'failed':15s}: {batch_stats['failed']:4d} ({batch_stats['failed']/total*100:.1f}%)")
            print(f"{'='*60}\n")
        
        return results
    
    def get_statistics(self):
        """Return geocoding statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0


# Example usage and testing
if __name__ == "__main__":
    print("🌍 Robust Geocoder - Test Suite\n")
    
    # Initialize geocoder for Sri Lanka
    geocoder = GeocodeFallback(country_code='LK', bounds=(5.9, 9.9, 79.5, 82.0))
    
    # Test 1: Single location
    print("=" * 60)
    print("Test 1: Single Location Geocoding")
    print("=" * 60)
    result = geocoder.geocode_location("Colombo", verbose=True)
    
    # Test 2: Batch locations
    print("\n" + "=" * 60)
    print("Test 2: Batch Geocoding (4 threads)")
    print("=" * 60)
    
    test_locations = [
        {'location': 'Colombo', 'district': 'Colombo'},
        {'location': 'Kandy', 'district': 'Kandy'},
        {'location': 'Galle', 'district': 'Galle'},
        {'location': 'Jaffna', 'district': 'Jaffna'},
        {'location': 'Trincomalee', 'district': 'Trincomalee'},
    ]
    
    results = geocoder.geocode_batch(test_locations, num_threads=4, verbose=True)
    
    print("\nFinal Results:")
    failed_count = 0
    for i, r in enumerate(results, 1):
        if r['success']:
            print(f"{i}. {r['original_location']:20s} → ({r['latitude']:.6f}, {r['longitude']:.6f}) via {r['source']}")
        else:
            print(f"{i}. {r['original_location']:20s} → FAILED")
            failed_count += 1

    # If there are failed locations and Serper API key is not set, print instructions
    if failed_count > 0 and not SERPER_API_KEY:
        print("\n---")
        print("Some locations failed to geocode. To improve accuracy by approximately 4-5%, you can enable the Serper provider (Google Search API).\n")
        print("To do this:")
        print("  1. Go to https://serper.dev/ and sign up for a free API key.")
        print("  2. Add a line to your .env file in this directory:")
        print("       SERPER_API_KEY=your_api_key_here\n")
        print("Then rerun your geocoding script.\n")
        print("Note: The 4-5% accuracy increase is an estimate based on typical fallback success rates; actual improvement may vary depending on your data.")
        print("---\n")
