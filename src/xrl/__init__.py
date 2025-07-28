from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xrl")
except PackageNotFoundError:
    __version__ = "unknown version"
