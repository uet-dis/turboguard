"""Dataset registry and per-dataset handlers.

Importing this package triggers auto-registration of all handlers.
"""

from datasets.unsw.handler import UNSWHandler
from datasets.kdd.handler import KDDHandler
from datasets.iec104.handler import IEC104Handler
from datasets.cic2018.handler import CIC2018Handler
from datasets.cic2017.handler import CIC2017Handler

__all__ = ["UNSWHandler", "KDDHandler", "IEC104Handler", "CIC2018Handler", "CIC2017Handler"]

