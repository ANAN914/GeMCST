from ._version import __version__

def __getattr__(name):
    if name == "GeMCSTRunner":
        from .project import GeMCSTRunner
        return GeMCSTRunner
    raise AttributeError(name)
