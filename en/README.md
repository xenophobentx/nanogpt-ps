# Watch a GPT continue text

In this model, one token is one character. At each step, the model gives
every known character a score. We turn the scores into probabilities,
select a character, and append it to the text.

You need PowerShell 7. Run the commands from the folder containing
`nanogpt-ps.ps1`. From the full repository root, first run
`Set-Location .\dist\en`. For a standalone package, start in its own folder.

## 1. Generate a short continuation

```powershell
pwsh -File .\nanogpt-ps.ps1 -Lang en -Prompt "ROMEO:" -MaxTokens 40
```

`ROMEO:` is the starting text. `MaxTokens 40` requests up to 40 new characters.
The script loads `nanogpt-shakespeare-weights.json` from its own folder.
`-Lang en` selects the interface language; it does not change the model's training.

Output can differ between runs because we sample from the probabilities.
A model this small often loses coherence in longer passages.

## 2. Inspect one character selection

```powershell
pwsh -File .\nanogpt-ps.ps1 -Lang en -Chat -MaxTokens 1
```

In the interface, enter:

```text
ROMEO:
/step 1
```

Entering `ROMEO:` already generates one character. `/step 1` shows the
selection of the character after that. Context includes your line,
an added newline, and the character already generated.

The most likely candidate does not always win. The selected character
enters the context, so another `/step 1` starts from a different input.
Use `/q` to exit.

```text
text -> character ids -> character and position vectors
     -> transformer layers -> final LayerNorm -> Head -> logits
     -> temperature + softmax -> top-k -> character selection
                                               |
                              next input <-----+

One transformer layer:
  LayerNorm -> attention -> add input
  LayerNorm -> MLP       -> add input
```

Logits are scores; softmax turns them into probabilities. Attention weights
control how value vectors from available positions are mixed. They are
different from next-character probabilities. `/step` shows selected results,
not every intermediate array.

## 3. Three short experiments

**Temperature: same input, different distribution.** Enter:

```text
/reset
/topk 0
/temp 0.3
/step 1
/reset
/temp 1.2
/step 1
```

With empty context, `/step` starts with a newline if the model knows it,
or character id 0 otherwise. Both parts start from the same input.
At lower temperature, larger probabilities dominate more strongly.
The selected character need not change for the distribution difference to be visible.

**Top-k: how many candidates remain eligible?** Enter:

```text
/reset
/temp 0.8
/topk 1
/step 1
```

One candidate remains. Its chance after filtering is 100%, even if its
probability before filtering was lower. Repeat `/reset` and `/step 1`:
with the same starting state and settings, sampling adds no variety.

Example: a=50%, b=30%, c=20%. With top-k=2, c is removed. The remaining
chances are 50/80=62.5% for a and 30/80=37.5% for b. Top-5 limits the
rows displayed; top-k limits which candidates can be selected.

**Context: does the preceding text matter?** In the same session:

```text
/reset
/topk 0
/temp 0.8
ROMEO:
/step 1
/reset
KING:
/step 1
```

Each `/step` shows the state after one character was generated automatically.
The candidate lists may differ because the preceding text differs.
This short experiment does not establish a fixed grammatical role for any head.

## 4. How does text become a training task?

```text
Text:      a l m a
Input:     a l m
Target:    l m a

Available text -> expected next character
             a -> l
            al -> m
           alm -> a
```

During training, the actual continuation is available. If the model gives `l`
a probability of 10% at the first position, the loss is `-ln(0.1)`, approximately
2.303. At 50%, it is approximately 0.693. Giving the correct continuation
more probability lowers the loss.

Backward computes how the loss responds to small weight changes. Adam uses
the gradients and moving averages from previous steps to update the weights.
Generation uses neither backward nor weight updates.

## 5. Run a small training demonstration

```powershell
pwsh -File .\nanogpt-ps.ps1 -Lang en -Train -Layers 1 -Dim 16 -Heads 2 -Block 16 -BatchSize 1 -Threads 1 -TrainSteps 20 -CorpusFile .\tinyshakespeare.txt -WeightsFile .\demo-weights.json
```

This creates a new model if `demo-weights.json` does not exist. Otherwise,
it resumes with that file's model dimensions. Choose a different filename
for a fresh experiment. `tinyshakespeare.txt` must be in the folder.

Twenty steps demonstrate the process; they are still within learning-rate
warmup. Do not expect readable text yet. Loss can fluctuate because it is
measured on different windows. Lower training loss alone does not tell you
how well the model performs on new text.

Check the hand-written backward separately:

```powershell
pwsh -File .\nanogpt-ps.ps1 -Lang en -GradCheck
```

This compares selected gradients with numerical estimates on a small model.
It does not measure text quality.

## 6. Read the code in this order

Search for these names:

1. `Invoke-Generate`: repeating character selection.
2. `Forward-Token`: processing one character.
3. `Sample-Logits`, `Sample-FromProbs`: selecting a character from scores.
4. `Compute-SeqGrad`, FORWARD: calculating training loss.
5. `Compute-SeqGrad`, BACKWARD: calculating gradients.
6. `Invoke-Train`: updating weights.

Screen drawing, saving, and MathNet acceleration can wait until later.
