from typing import List, Set

from core.models import PipelineSentence


class RelevantInformationExtractor:

    def __init__(self):
        pass

    def extract(self, sentences: List[PipelineSentence], window=0) -> List[PipelineSentence]:
        """
        Extract the most relevant sentences given annotated sentences with relevance.
        The sentences should not be reordered previously.
        There cannot be duplicate sentences - if two windows overlap, the second window will not contain the overlap
        - window is the size of a scanning window, specifying a non-zero value will cause the sentences around a
          relevant sentence to be bundled together. Each bundle will have size 2*window + 1
        """
        relevances_and_indices = []  # Relevance and index tuples
        for idx, sentence in enumerate(sentences):
            relevances_and_indices.append((sentence.relevance, idx))
        relevances_and_indices.sort(key=lambda x: x[0], reverse=True)

        sorted_sents: List[PipelineSentence] = []
        added_indices: Set[int] = set()
        while relevances_and_indices:
            # Break early if we've added everything
            if len(added_indices) == len(sentences):
                break
            _, sent_index = relevances_and_indices.pop(0)
            # Already added, skip
            if sent_index in added_indices:
                continue
            min_idx = max(0, sent_index - window)
            max_idx = min(len(sentences), sent_index + window + 1)
            for idx in range(min_idx, max_idx):
                if idx in added_indices:
                    # Sentence added already, skip
                    continue
                sorted_sents.append(sentences[idx])
                added_indices.add(idx)

        return sorted_sents
