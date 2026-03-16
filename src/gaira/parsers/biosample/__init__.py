"""Biosample dataset parser scaffolds."""

from gaira.parsers.biosample.base import BiosampleParserBase
from gaira.parsers.biosample.shine_ev_sers_parser import ShineEVSERSParser

__all__ = ["BiosampleParserBase", "ShineEVSERSParser"]
