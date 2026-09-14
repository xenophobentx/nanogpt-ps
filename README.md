[Magyarul: README.hu.md](README.hu.md)

# psgpt

A character-level GPT written in PowerShell, with nothing underneath it. The transformer is in the script: token and position embeddings, multi-head causal self-attention, an MLP with GELU, LayerNorm, the output head. So is the training loop. Backpropagation is written out by hand and checked against numerical gradients to about 1e-9, the optimizer is Adam with a cosine learning-rate schedule, and mini-batches run in parallel across your CPU cores. Generation uses a KV-cache. There is a chat window in the terminal.

The reason to care is that you can read all of it. Every matrix multiply is a loop you can put a breakpoint on. And one command in the chat, `/step`, turns that into something you can watch: it produces the next character and prints what happened on the way there.

## What you can do with it

Chat. Type a line and the model continues it in the style it was trained on:

```
  > ROMEO:
  ◆ KING RICHARD II:
    And thou consent is so this first wivers,
    For thou art thou only to cheek the corruption,
```

Watch it decide. `/step` generates one character and shows the pipeline for that character: the token plus position embedding, which earlier characters each layer's attention weighted most, then the final logits, the softmax over them, the five most likely characters with their probabilities, and the one that was actually sampled. `/step 5` does five in a row. You are looking at the model choose a letter.

Train. `nanogpt-ps.ps1 -Train` runs the forward pass, the hand-written backward pass and Adam over a text file of your choosing (`-CorpusFile`), checkpointing to a weights file as it goes. `-GradCheck` runs the numerical gradient check on a tiny model, which is how you convince yourself the backprop is right.

## Quickstart

The weight files are about 57 MB each, so they are published under GitHub Releases rather than committed to the repo. Download `nanogpt-shakespeare-weights.json` and `nanogpt-orban-weights.json` into the edition folder you are going to use.

```
cd en          # or: cd hu
# put the two weight .json files from Releases in this folder
pwsh ./chat.ps1 -Model shakespeare      # or: -Model orban
```

Type something and press Enter. Inside the chat:

- `/step`, or `/step 5`: generate character by character with the full breakdown
- `/temp`, `/tokens`, `/topk`: sampling temperature, output length, top-k cutoff
- `/help`: everything else
- `/q`: quit

## The two models

Both have 6 layers, a width of 192, about 2.7 million parameters, and a context window of 128 characters.

`shakespeare` was trained on Tiny Shakespeare, roughly 1 MB of the plays. It writes English in the shape of a script: speaker names in capitals, line breaks where the verse would put them, words that are mostly real and sometimes nearly so.

`orban` was trained on Hungarian political speeches. Its output is a style simulation: invented sentences with the rhythm and vocabulary of the source. Nothing it says is a quotation. The model has no record of any actual speech and no way to reproduce one; it emits plausible Hungarian one character at a time. The chat window labels the output as a simulation for exactly this reason, and that is how it should be read.

## Speed and scope

It is slow. Character-level, interpreted, PowerShell: about 36 characters a second on a laptop, around 84 on a faster desktop with the native library loaded. A paragraph takes a while to appear. Training a model the size of the bundled ones, on CPU, is a multi-day job.

It is also small. 2.7 million parameters and 128 characters of context are enough to learn spelling, punctuation and the cadence of a text, and little beyond that. It will not answer questions. Treat it as a GPT you can read end to end, with the generated text as evidence that the reading was correct.

## Requirements

PowerShell 7 on Windows, Linux or macOS. No Python, no ML framework, no GPU.

`lib/MathNet.Numerics.dll` ships alongside the scripts and is optional. With it present, generation runs about 8x faster. Without it, everything still works on the pure-PowerShell path.

## en/ and hu/

The repo carries two copies of the code: `en/` with English comments, `hu/` with Hungarian ones. The code is identical; pick the folder whose comments you would rather read, and put the weight files there.

## License

MIT. See the LICENSE file next to this one.
