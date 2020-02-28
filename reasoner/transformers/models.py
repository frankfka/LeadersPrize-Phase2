from enum import Enum


class Entailment(Enum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class StsSimilarity(Enum):
    EQUIV = "equivalent"
    MOSTLY_EQUIV = "most"
    ROUGH_EQUIV = "roughly_equivalent"
    NOT_EQUIV_SHARE_DETAIL = "not_equiv_share_detail"
    NOT_EQUIV_SAME_TOPIC = "not_equiv_same_topic"
    DISSIMILAR = "dissimilar"
