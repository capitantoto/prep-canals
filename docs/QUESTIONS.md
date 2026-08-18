# STUDY QUESTIONS


1. Your split stratifies by canonical, so every catalog SKU has training pairs. For which of the three rungs does that make the eval optimistic — TF-IDF, zero-shot,
fine-tuned? (Hint: which of the three ever sees the pairs?) What second split would you build to measure "cold-start" SKUs, and which sklearn splitter groups by y?

For TF-IDF, whose vocabulary was built exclusively from existing training pairs, and probably ever so slightly for the fine-tuned version. zero-shot will be at a disadvantage at elast in theory.

To measure cold-start SKUs, I'd first randomly sample a percentage _if the unique response/catalog SKUs_ (the unique values in y) -- say 5-20% -- and excluide all of the X samples for those SKUs from the traiing data. This should be enough to measure the cold start problem, but in fact finer distinctions between SKUs might be needed (it's easier to correctly cold-start 'red electrical tape' if 'black electrical tape' is in the training data than if it isn't) to do 'whole-product cold-start' measurements if a compeltely new catgegory shows up instrad of a enw vraint on an existing one.

2. Predict, in writing, whether zero-shot MiniLM beats your char-trigram TF-IDF here. My money says it loses — reason about what "thhn cu 12ga sol rd" looks like to a
model pretrained on natural English vs. to a character-trigram bag. This prediction-then-result is the spine of the blog post either way.

I'd believe MiniLM loses without normalization and wins with it.

3. Should normalize() run before embedding, too? What are you measuring if you leave it on vs. take it off? (There's no wrong answer, only an uncontrolled experiment
if you don't decide consciously.)

I think I should normalize, too. There is no semantic difference between "sol rd" and "solid red" but tokens _will_ differ. So any ambigüity we can manually remove before embedding should help or at least not hurt.
