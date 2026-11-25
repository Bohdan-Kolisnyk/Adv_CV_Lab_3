# Hypotheses log

Every block of experiments gets its hypothesis and predicted outcome here before the first run. The result and the explanation are added after the block.

## Block 1: alignment of the baseline and our connector (Part 2)

Setup (identical for both versions): frozen SigLIP 2 B/16-512 encoder, frozen SmolLM2-360M-Instruct, only the connector is trained, Flickr30k Karpathy train split, same number of steps, same batch size, same learning-rate schedule, same seed. Loss on the answer tokens only.

- Baseline: pixel shuffle k=4 + linear layer, 64 image tokens.
- Ours: pixel shuffle k=2 + 2-layer MLP, 256 image tokens.

### H1.1 Caption quality
Hypothesis: our connector is not clearly better than the baseline on captions.
Predicted outcome: val caption loss lower by 0.01 to 0.04 for ours, BLEU-4 higher by 0 to 0.02, both inside the noise of a single seed.
Why: in the Lecture 1 experiment with a linear connector, k=2 versus k=4 gave test loss 2.228 versus 2.246 and BLEU-4 0.274 versus 0.269. Captions mostly depend on what is in the image, not on where it is, so extra tokens and an MLP should add little.

### H1.2 Visual gaps
Hypothesis: both versions use the image content but ignore token order.
Predicted outcome: swap gap (another image) clearly positive for both, around 1.0 to 1.3 nats; permuted-token gap close to 0 for both (below 0.05); zeros gap positive.
Why: in Lecture 1 and Lecture 2 the permuted-token gap was about 0 for projection connectors. A caption loss does not need layout. This is also the reason to expect that captioning quality will not predict how well the model finds objects.

### H1.3 Text-only behaviour
Hypothesis: the language model keeps its text-only behaviour exactly.
Predicted outcome: text KL against the original SmolLM2 is 0 up to numerical noise for both versions.
Why: the language model is frozen and there are no LoRA weights, so a prompt without image tokens runs through identical weights.

### H1.4 Cost
Hypothesis: our version is slower per training step.
Predicted outcome: 1.5x to 2x step time compared with the baseline, same encoder cost.
Why: the sequence through the language model grows from about 90 to about 290 tokens, while the encoder still processes 1024 patches in both versions.

### Result
To be filled in after the block.
