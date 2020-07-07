from typing import List, Set, Optional

from analyze.sentence_relevance_scorer.word2vec_relevance_scorer import Word2VecRelevanceScorer
from core.models import PipelineSentence


class RelevantInformationExtractor:

    def __init__(self, sentence_relevance_scorer: Word2VecRelevanceScorer):
        self.sentence_relevance_scorer = sentence_relevance_scorer

    def extract_for_transformer(
            self, claim_str: str, supporting_sentences: List[PipelineSentence],
            min_sent_relevance: float, deduplication_relevance_cutoff: float, max_num_words: float
    ) -> List[PipelineSentence]:
        """
        Extract a set of sentences to be used by the transformer
        """
        # Enforce a minimum relevance and sort by decreasing relevance
        sents_for_extraction = [sent for sent in supporting_sentences if sent.relevance > min_sent_relevance]
        sents_for_extraction = sorted(sents_for_extraction, key=lambda sentence: sentence.relevance, reverse=True)
        # Now we need to deduplicate to remove sentences that are highly similar
        extracted_sents: List[PipelineSentence] = []
        rough_num_words = 0
        for sent in sents_for_extraction:
            if rough_num_words > max_num_words:
                break
            # Ignore those very similar with claim
            if self.sentence_relevance_scorer.get_relevance(claim_str, sent.preprocessed_text) > \
                    deduplication_relevance_cutoff:
                continue
            # Ignore those with high similarity with a sentence already extracted
            # Note: we only need to check the last one because these sentences are ordered by relevance
            if len(extracted_sents) > 0 and \
                    self.sentence_relevance_scorer.get_relevance(
                        sent.preprocessed_text,
                        extracted_sents[-1].preprocessed_text
                    ) > deduplication_relevance_cutoff:
                continue
            # We should extract this, passed all checks
            extracted_sents.append(sent)
            rough_num_words += len(sent.preprocessed_text.split())
        return extracted_sents

    def extract_with_window(self, sentences: List[PipelineSentence], left_window=0, right_window=0) -> List[PipelineSentence]:
        """
        Extract the most relevant sentences given annotated sentences with relevance.
        The sentences should not be reordered previously.
        - There cannot be duplicate sentences - if two windows overlap, the second window will not contain the overlap
        - Left and right windows specify the size of a scanning window, specifying a non-zero value will cause the
          sentences around a relevant sentence to be bundled together.
        - Each bundle will have max size left_window + right_window + 1
        """
        article_url: Optional[str] = None
        relevances_and_indices = []  # Relevance and index tuples
        for idx, sentence in enumerate(sentences):
            if not article_url:
                # Populate parent article URL
                # IMPORTANT: This assumes that we call `extract` once per article
                article_url = sentence.parent_article_url
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
            sentence_strs: List[(str, str)] = []  # Tuples of (original, preprocessed)
            max_relevance: float = 0
            for idx in range(min_idx, max_idx):
                if idx in added_indices:
                    # Sentence added already, skip
                    continue
                sentence = sentences[idx]
                sentence_strs.append((sentence.text, sentence.preprocessed_text))
                if sentence.relevance > max_relevance:
                    max_relevance = sentence.relevance
                added_indices.add(idx)
            # Add the extracted block to the return result
            if sentence_strs:
                # Create a new pipeline sentence, with relevance that is the maximum relevance of its subsentences
                original_sentence_strs = []
                preprocessed_sentence_strs = []
                for (original, preprocessed) in sentence_strs:
                    original_sentence_strs.append(original)
                    preprocessed_sentence_strs.append(preprocessed)
                block_sentence_original = " | ".join(original_sentence_strs)
                block_sentence_preprocessed = " | ".join(preprocessed_sentence_strs)
                # Create a pipeline sentence for the block
                block_pipeline_sentence = PipelineSentence()
                block_pipeline_sentence.text = block_sentence_original
                block_pipeline_sentence.preprocessed_text = block_sentence_preprocessed
                block_pipeline_sentence.parent_article_url = article_url
                block_pipeline_sentence.relevance = max_relevance
                extracted.append(block_pipeline_sentence)

        return extracted
