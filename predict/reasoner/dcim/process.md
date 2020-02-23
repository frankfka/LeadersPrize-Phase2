Train models on the SNLI dataset using default parameters and an adam optimizer.
```python
parser.add_argument("--learning_rate", type=float, default=0.0004, help="Learning rate for model")
parser.add_argument("--keep_rate", type=float, default=0.5, help="Keep rate for dropout in the model")
parser.add_argument("--seq_length", type=int, default=50, help="Max sequence length")
```
The training continues until improvement is under a threshold.

Supply cmd parameters <model type> <trained model name> such as 'esim train_snli_esim'

Load the pretrained word indices and embeddings and  give them to the classifier model.

Articles, assumed as premises, are from the LT,a nd are loaded from the input files by extracting text from the html.
This is very small set to test drive part of dcim model, our model.

Html elements containing at least 1 space, no semicolons, at least 5 characters, and not belonging to a known meta field such as 'script' or 'style' are extracted as text.

Each article is then split on the '.' symbol into sentences, and sentences less than 5 characters are removed.

For each article, we make pairs where each sentence is a premise, and the claim is the hypothesis.

Run the model on each each pair to get classify whether the premise entails, contradicts, or is neutral to the hypothesis, that is, the claim. Thes are teh first three tasks that the dcim should hanlde.


Give the article a score of (#entailments - #contradictions) / #sentences. 

High scores that fall between 0.3 and 1 suggest entailments, while negative scores between -0.3 t0 -1 suggest contradiction, and scores that fall between -0.3 to +0.3 can be labelled as neutral.

