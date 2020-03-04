from typing import List, Set

from core.models import PipelineSentence


class RelevantInformationExtractor:

    def extract(self, sentences: List[PipelineSentence], left_window=0, right_window=0) -> List[PipelineSentence]:
        """
        Extract the most relevant sentences given annotated sentences with relevance.
        The sentences should not be reordered previously.
        - There cannot be duplicate sentences - if two windows overlap, the second window will not contain the overlap
        - Left and right windows specify the size of a scanning window, specifying a non-zero value will cause the
          sentences around a relevant sentence to be bundled together.
        - Each bundle will have max size left_window + right_window + 1
        """
        relevances_and_indices = []  # Relevance and index tuples
        for idx, sentence in enumerate(sentences):
            relevances_and_indices.append((sentence.relevance, idx))
        relevances_and_indices.sort(key=lambda x: x[0], reverse=True)

        extracted: List[PipelineSentence] = []
        added_indices: Set[int] = set()  # Sentence indices that have already been extracted
        while relevances_and_indices:
            # Break early if we've added everything
            if len(added_indices) == len(sentences):
                break
            _, sent_index = relevances_and_indices.pop(0)
            # Already added, skip
            if sent_index in added_indices:
                continue
            # Calculate sentence indices for iteration
            min_idx = max(0, sent_index - left_window)
            max_idx = min(len(sentences), sent_index + right_window + 1)
            # Contains extracted sentences for this block, should have max length left_window + right_window + 1
            sentence_strs: List[str] = []
            max_relevance: float = 0
            for idx in range(min_idx, max_idx):
                if idx in added_indices:
                    # Sentence added already, skip
                    continue
                sentence = sentences[idx]
                sentence_strs.append(sentence.sentence)
                if sentence.relevance > max_relevance:
                    max_relevance = sentence.relevance
                added_indices.add(idx)
            # Add the extracted block to the return result
            if sentence_strs:
                # Create a new pipeline sentence, with relevance that is the maximum relevance of its subsentences
                block_sentence = " ".join(sentence_strs)
                block_pipeline_sentence = PipelineSentence(block_sentence)
                block_pipeline_sentence.relevance = max_relevance
                extracted.append(block_pipeline_sentence)

        return extracted
