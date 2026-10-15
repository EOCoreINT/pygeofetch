"""
PyGeoFetch core package.

Provides the main PyGeoFetch engine plus authentication, search,
download, scheduling, and cache management subsystems.
"""

from pygeofetch.core.authenticator import AuthManager
from pygeofetch.core.downloader import AdaptiveDownloader
from pygeofetch.core.engine import PyGeoFetch
from pygeofetch.core.searcher import FederatedSearcher

__all__ = [
    "PyGeoFetch",
    "AuthManager",
    "FederatedSearcher",
    "AdaptiveDownloader",
]
