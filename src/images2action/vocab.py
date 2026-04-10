"""
Shared RDF namespaces and controlled vocabularies for the vision KG.
"""

from rdflib import Namespace

CV = Namespace("http://vision.semkg.org/onto/v0.1/")
SCHEMA = Namespace("http://schema.org/")
#EX_BDD = Namespace("http://example.org/bdd/")
#EX_COCO = Namespace("http://example.org/coco/")
EX = Namespace("http://example.org/instance/")

# Traffic-light colors
COLOR_MAP = {
    "red": CV.RedColor,
    "green": CV.GreenColor,
    "yellow": CV.YellowColor,
}
VALID_TL_COLORS = set(COLOR_MAP.keys())

# Ego vehicle actions
ACTION_MAP = {
    "stop": CV.StopAction,
    "go": CV.GoAction,
    "slow": CV.SlowAction,
    "unknown": CV.UnknownAction,
}
