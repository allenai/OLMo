import csv
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy

from .extrapolate_d import get_coefficients


@dataclass
class CurveFitConfig:
    path: str
    """
    Path containing the W&B downloaded data and metadata.
    """

    keys: List[str]
    """
    The metrics for computing the scaling law predictions.
    """

    n: int
    """
    The number of parameters in the model.
    """

    mode: str
    """
    Whether this model is used for fitting the curve ('train') or evaluating the fit ('eval').
    """

    label: str
    """
    A short label for this curve.
    """

    color: str
    """
    The color for this curve.
    """

