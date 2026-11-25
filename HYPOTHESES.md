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
Run: 2000 steps, batch 16, Tesla T4, fp16 autocast, one seed. Validation is the Flickr30k Karpathy val split (1014 images).

| | baseline (k=4, linear) | ours (k=2, MLP) |
|---|---|---|
| image tokens | 64 | 256 |
| trainable parameters | 11,796,480 | 3,871,680 |
| train time, s | 1555 | 2577 |
| peak memory, GB | 4.87 | 7.84 |
| val loss | 2.4126 | 2.3914 |
| BLEU-4 | 0.2659 | 0.2654 |
| gap zeros | 1.245 | 1.644 |
| gap noise | 1.912 | 1.944 |
| gap swap | 1.132 | 1.123 |
| gap permuted | 0.002 | 0.009 |
| gap no image | 1.181 | 1.201 |
| image/text norm ratio | 257.7 | 91.3 |

Paired bootstrap over images (200 resamples), BLEU-4 ours minus baseline: 95% CI [-0.0113, 0.0123].

**H1.1: partly confirmed.** Val loss is lower for ours by 0.021, inside the predicted 0.01 to 0.04. BLEU-4 is lower by 0.0005, not higher by 0 to 0.02 as predicted, and the confidence interval of the difference contains 0. So there is no measurable caption-quality difference. The validation curves show that the baseline learns faster (ours is behind until about step 1250) and ours ends slightly lower only in the last 500 steps. With one seed we do not claim that ours is better; the selection rule (lower val loss) picks ours.

**H1.2: confirmed.** Swap gap is 1.13 and 1.12, permuted gap is 0.002 and 0.009, zeros gap is positive for both. Both models use the content of the image and ignore the order of the image tokens. The two connectors behave the same on swap, permuted, noise and no-image gaps; only the zeros gap differs (1.64 versus 1.24), which we read as a larger reaction to an out-of-distribution input, not as more use of the image.

**H1.3: not measured in this block.** The language model is frozen and there are no LoRA weights, so a prompt without image tokens runs through identical weights and the KL is 0 by construction. The measurement is deferred to the fine-tuning block, where the language model weights change.

**H1.4: confirmed.** Training time is 1.66x the baseline, inside the predicted 1.5x to 2x. Peak memory is 1.6x. Ours has three times fewer trainable parameters, because the baseline linear layer maps 12,288 inputs to 960 outputs while ours maps 3,072 to 960 and adds a 960 to 960 layer.

**Not predicted: image token norm.** The mean norm of image tokens is 258 times (baseline) and 91 times (ours) the mean norm of text embeddings. The connector output is far larger than anything the language model saw in pretraining, and it is smaller for ours. We did not use the output norm layer of the connector; if fine-tuning is unstable, this is the first thing to check.
