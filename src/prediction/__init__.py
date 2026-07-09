"""Inference-only prediction engine for the football result model.

Public surface:
    Predictor            - match / batch prediction
    FeatureBuilder       - reconstruct the exact training feature vector
    MonteCarloSimulator  - seeded Monte Carlo over model probabilities
    WorldCupSimulator    - full tournament simulation -> champion probabilities
    loaders              - cached artifact/reference loading
    validation           - typed exceptions and contract checks
    utils                - seeding, normalisation, name resolution
"""
from src.prediction.predictor import Predictor
from src.prediction.feature_builder import FeatureBuilder
from src.prediction.simulator import MonteCarloSimulator
from src.prediction.tournament import WorldCupSimulator
from src.prediction.worldcup2026 import WorldCup2026
from src.prediction import loaders, validation, utils, scoreline, betting

__all__ = ["Predictor", "FeatureBuilder", "MonteCarloSimulator",
           "WorldCupSimulator", "WorldCup2026", "loaders", "validation", "utils",
           "scoreline", "betting"]
