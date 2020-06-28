import dataclasses
from enum import Enum

import numpy as np


class Prediction(Enum):
    TRUE = "TRUE"
    NEUTRAL = "NEUTRAL"
    FALSE = "FALSE"


@dataclasses.dataclass
class BeliefAnalysis:
    entailment: float
    neutral: float
    contradiction: float
    paraphrase: float
    evidence: float
    asserting: float
    hedging: float
    questioning: float
    disagreeing: float
    stancing: float
    negative: float
    fakeness: float
    relevance: float

    @property
    def true_classification(self) -> Prediction:
        close_tolerance = 0.08

        if np.isclose(self.true_magnitude, self.neutral_magnitude, rtol=close_tolerance):  # step 7iii
            return Prediction.NEUTRAL
        elif np.isclose(self.false_magnitude, self.neutral_magnitude, rtol=close_tolerance):  # step 7iv
            return Prediction.NEUTRAL
        elif np.isclose(self.true_magnitude, self.false_magnitude, rtol=close_tolerance):  # step 7v
            return Prediction.NEUTRAL
        elif self.true_magnitude > self.false_magnitude:  # step 7vi
            return Prediction.TRUE
        elif self.true_magnitude < self.false_magnitude:  # step 7vii
            return Prediction.FALSE
        else:
            assert False

    @property
    # @functools.lru_cache() #todo this causes a hash error, but may be a good performance increase if fixed
    def true_magnitude(self) -> float:
        return np.linalg.norm(self.true_attribs) / np.linalg.norm([1] * len(self.true_attribs))

    @property
    # @functools.lru_cache()
    def neutral_magnitude(self) -> float:
        return np.linalg.norm(self.neutral_attribs) / np.linalg.norm([1] * len(self.neutral_attribs))

    @property
    # @functools.lru_cache()
    def false_magnitude(self) -> float:
        return np.linalg.norm(self.false_attribs) / np.linalg.norm([1] * len(self.false_attribs))

    @property
    def true_attribs(self):
        return np.array((self.entailment, self.relevance, self.asserting, self.evidence,
                         # self.support
                         ))

    @property
    def neutral_attribs(self):
        return np.array((self.paraphrase, self.questioning, self.disagreeing, self.hedging, self.neutral))

    @property
    def false_attribs(self):
        return np.array((self.negative, self.contradiction, self.stancing, self.fakeness,
                         # self.deny
                         ))

    @classmethod
    def from_dict(cls, data):
        entailment = data['entailment']
        evidence = data['evidence']
        asserting = data['asserting']
        paraphrase = data['paraphrase']
        hedging = data['hedging']
        questioning = data['questioning']
        neutral = data['neutral']
        disagreeing = data['disagreeing']
        contradiction = data['contradiction']
        stancing = data['stancing']
        negative = data['negative']
        fakeness = data['fakeness']
        relevance = data['relevance']

        return cls(
            entailment=float(entailment), evidence=float(evidence),

            asserting=float(asserting), paraphrase=float(paraphrase), hedging=float(hedging),
            questioning=float(questioning), neutral=float(neutral),

            disagreeing=float(disagreeing), contradiction=float(contradiction), stancing=float(stancing),
            negative=float(negative), fakeness=float(fakeness),

            relevance=relevance
        )
