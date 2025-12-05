from setuptools import setup, find_packages

setup(
    name="robust-geocoder",
    version="0.1.0",
    description="A comprehensive geocoding library with multiple fallback providers and postal code lookup.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/geocode-fallback-library",
    py_modules=["geocode_fallback_library"],
    install_requires=[
        "pandas",
        "pgeocode",  # still required for postal code lookup
        "requests",
        "python-dotenv",
        "geopy",
        "concurrent-futures; python_version<'3.2'",
        "SPARQLWrapper; extra == 'wikidata'",
        "duckduckgo-search; extra == 'duckduckgo'"
    ],
    extras_require={
        "wikidata": ["SPARQLWrapper"],
        "duckduckgo": ["duckduckgo-search"]
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "geocode-fallback-test=geocode_fallback_library:main"
        ]
    },
)