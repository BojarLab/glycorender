from .render import *

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("glycorender")
except PackageNotFoundError:  # running from a source checkout that was never pip-installed
    __version__ = "0.0.0+local"
