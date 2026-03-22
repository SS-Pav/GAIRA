"""Biosample dataset parser scaffolds."""

from gaira.parsers.biosample.base import BiosampleParserBase
from gaira.parsers.biosample.covid_serum_raman_parser import COVIDSerumRamanParser
from gaira.parsers.biosample.cspp_serum_parser import CSPPSerumParser
from gaira.parsers.biosample.diabetes_plasma_ev_sers_parser import DiabetesPlasmaEVSERSParser
from gaira.parsers.biosample.ergothioneine_serum_parser import ErgothioneineSerumParser
from gaira.parsers.biosample.hcc_serum_parser import HCCSerumParser
from gaira.parsers.biosample.serum_ag_colloids_parser import SerumAgColloidsParser
from gaira.parsers.biosample.serum_protocol_comparison_parser import SerumProtocolComparisonParser
from gaira.parsers.biosample.shine_ev_sers_parser import ShineEVSERSParser
from gaira.parsers.biosample.small2023_ev_parser import Small2023EVParser

__all__ = [
    "BiosampleParserBase",
    "COVIDSerumRamanParser",
    "CSPPSerumParser",
    "DiabetesPlasmaEVSERSParser",
    "ErgothioneineSerumParser",
    "HCCSerumParser",
    "SerumAgColloidsParser",
    "SerumProtocolComparisonParser",
    "ShineEVSERSParser",
    "Small2023EVParser",
]
