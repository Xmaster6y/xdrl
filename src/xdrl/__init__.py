from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xdrl")
except PackageNotFoundError:
    __version__ = "unknown version"
