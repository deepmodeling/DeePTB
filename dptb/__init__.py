try:
    import importlib.metadata as importlib_metadata
except ImportError:
    # for Python 3.7 
    import importlib_metadata

try:
    __version__ = importlib_metadata.version("dptb")
except importlib_metadata.PackageNotFoundError:
    __version__ = "unknown"