[Magyarul: README.hu.md](README.hu.md)

# psgpt

A character-level GPT in PowerShell, based on nanoGPT. The whole transformer is in the script, and there is a chat window in the terminal.

Why? Because it runs on plain PowerShell. No Python, nothing else.

Every matrix multiply is a loop you can put a breakpoint on. The chat also has a `/step` command that generates the next character and prints what happened along the way.

![The nanoGPT-ps chat window: the model header, then a Shakespeare-style continuation](docs/chat.png)

*The chat window: the model header, then a Shakespeare-style continuation.*

## What you can do with it

Chat. Type a line and the model continues it in the style it was trained on:

```
  > ROMEO:
  ◆ KING RICHARD II:
    And thou consent is so this first wivers,
    For thou art thou only to cheek the corruption,
```

Watch it decide. `/step` generates one character and shows the pipeline for that character: the token plus position embedding, which earlier characters each layer's attention weighted most, then the final logits, the softmax over them, the five most likely characters with their probabilities, and the one that was actually sampled. `/step 5` does five in a row.

Train. `nanogpt-ps.ps1 -Train` runs the forward pass, the hand-written backward pass and Adam over a text file of your choosing (`-CorpusFile`), and saves checkpoints to a weights file as it goes. `-GradCheck` runs a numerical gradient check on a tiny model, so you can see for yourself that the backprop is right.

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

`orban` was trained on Hungarian political speeches. Its output is a style simulation: made-up sentences with the rhythm and vocabulary of the source, generated one character at a time. None of it is a quote, and the model has no way to reproduce an actual speech. The chat window labels the output as a simulation.

## Speed and scope

It is slow. A character-level model interpreted in PowerShell does about 36 characters a second on a laptop, and around 84 on a faster desktop with the native library loaded. A paragraph takes a while to appear. Training a model the size of the bundled ones on CPU takes several days.

It is also small. 2.7 million parameters and 128 characters of context are enough to learn spelling, punctuation and the cadence of a text, and not much more. It will not answer questions. The point is that you can read the whole thing, end to end.

## Requirements

PowerShell 7 on Windows, Linux or macOS. No Python, no ML framework, no GPU.

`lib/MathNet.Numerics.dll` ships alongside the scripts and is optional. With it, generation runs about 8x faster. Without it, everything still works on the pure PowerShell path.

## en/ and hu/

The repo has two copies of the code: `en/` with English comments, `hu/` with Hungarian ones. The code is identical. Pick the folder whose comments you'd rather read, and put the weight files there.

## License

MIT. See the LICENSE file next to this one.
