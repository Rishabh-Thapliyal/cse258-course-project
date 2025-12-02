"""
LLM-based Recommender System

A collaborative filtering-based recommendation system that uses a frozen LLM backbone
to learn user and item embeddings for sequential recommendation.

Inspired by CLLM4Rec and related work.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .models import *
from .data import *
from .trainers import *

