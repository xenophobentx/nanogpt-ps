<#
.SYNOPSIS
    A GPT that continues text one character at a time, written in PowerShell.
.DESCRIPTION
    The model takes a text fragment and estimates the probability of each
    possible next character. We select a character, append it to the text,
    and repeat. In this model, one token is one character.

    During training, the actual continuation is available. We use it to
    compute the loss and update the model's trainable numbers, or weights.
    During generation, the weights stay fixed.

    To follow the source, you should be familiar with PowerShell arrays,
    loops, and functions. GPT concepts are introduced beside the relevant
    calculations. Running the script requires PowerShell 7.

    Start by generating text with an existing weight file. Training in
    PowerShell is slow; a few steps on a small model are enough to begin
    exploring how it works.

    No external library is required. During generation, the bundled MathNet
    library can speed up the calculations; -NoFast disables it.
.PARAMETER Train
    Start training. Resume from the weight file if it exists.
    Load the saved Adam state when available.
.PARAMETER TrainSteps
    Number of training steps to run in this invocation. Default: 3000.
.PARAMETER BatchSize
    Number of text windows used for one weight update. Default: 8.
.PARAMETER Threads
    Maximum number of parallel tasks. Defaults to the logical processor
    count reported by .NET. BatchSize is a separate setting.
.PARAMETER GradCheck
    Compare the hand-written gradient with a numerical estimate on
    a small model, checking four selected elements per parameter group.
.PARAMETER Layers
    Number of transformer layers in a new model. Default: 3.
.PARAMETER Dim
    Length of the vector at each position in a new model. Default: 64.
    Must be divisible by Heads.
.PARAMETER Heads
    Number of attention heads per layer in a new model. Default: 4.
.PARAMETER Block
    Maximum text window length in a new model. Default: 64.
    When resuming, model dimensions come from the weight file.
.PARAMETER Lang
    Interface language: hu or en. Does not change the language the model learned.
.PARAMETER MaxTokens
    Maximum number of new characters to generate. Default: 200.
.PARAMETER Temperature
    A positive number that adjusts the sampling distribution.
    Below 1, larger probabilities dominate; above 1, probabilities become closer.
.PARAMETER TopK
    Sample only from the K most likely characters. 0 keeps all characters.
.PARAMETER WeightsFile
    Path to the weight file to load, or to save during training.
.PARAMETER CorpusFile
    Path to the text file used for training.
.EXAMPLE
    pwsh -File .\nanogpt-ps.ps1 -Lang en -Prompt "ROMEO:" -MaxTokens 80
.EXAMPLE
    pwsh -File .\nanogpt-ps.ps1 -Lang en -Chat
    In the interface, /step 1 shows how one character is selected.
.EXAMPLE
    pwsh -File .\nanogpt-ps.ps1 -Lang en -GradCheck
.EXAMPLE
    pwsh -File .\nanogpt-ps.ps1 -Lang en -Train -Layers 1 -Dim 16 -Heads 2 -Block 16 -BatchSize 1 -Threads 1 -TrainSteps 20 -WeightsFile .\demo-weights.json
    Creates a small model if demo-weights.json is new; otherwise resumes from it.
#>
param(
    [string]$Prompt = '',
    [int]$MaxTokens = 200,
    [double]$Temperature = 0.8,
    [int]$TopK = 0,
    [ValidateSet('hu', 'en')][string]$Lang = 'hu',
    [switch]$Endless,
    [switch]$Chat,
    [switch]$AsLibrary,
    [switch]$NoFast,
    [switch]$Train,
    [int]$TrainSteps = 3000,
    [int]$BatchSize = 8,
    [int]$Threads = [Environment]::ProcessorCount,
    [double]$LearningRate = 1e-3,
    [int]$CheckpointEvery = 100,
    [switch]$GradCheck,
    [int]$Layers = 3,
    [int]$Dim = 64,
    [int]$Heads = 4,
    [int]$Block = 64,
    [string]$CorpusFile = (Join-Path $PSScriptRoot 'tinyshakespeare.txt'),
    [string]$WeightsFile = (Join-Path $PSScriptRoot 'nanogpt-shakespeare-weights.json')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:Lang = $Lang   # 'hu' or 'en' - the language of the chat interfaces
# English mode uses a '.' decimal point everywhere (numbers in gradcheck, training and generation),
# not just in the chat UI, so the documented -Lang en commands read the same on any locale.
if ($script:Lang -eq 'en') { try { [Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::InvariantCulture } catch { } }

<#
  Reading route

  1. Text continuation: Invoke-Generate.
     Follow how the selected character becomes the next input.
  2. Processing one character: Forward-Token.
     Embedding -> transformer layers -> final LayerNorm -> Head -> logits.
  3. Selecting a character: Sample-Logits and Sample-FromProbs.
     Logits -> temperature + softmax -> top-k -> sampling.
  4. Training: Compute-SeqGrad, starting with FORWARD only.
     The same model, but now the actual next character is available.
  5. Updating weights: BACKWARD, then Invoke-Train and its Adam block.

  Search for these function names to read the file in this order.
  Saving, parallel execution, and screen drawing can wait until later.

  Notation:
    token / id: a character in this model, and its integer identifier;
    vocab: number of known characters, including spaces and newlines;
    d / Dim: number of elements in each position's vector;
    T / seqLen: number of input positions processed together;
    nH / Heads: number of attention heads; hd = d / nH;
    nL / Layers: number of transformer layers;
    P: trainable parameter tables; G: their gradients.

  A vector is a sequence of numbers; a matrix is a table of rows and columns.
  The code stores matrices as one-dimensional arrays:
  M[i,j] is stored at A[i*columnCount+j]. Indices start at 0.
#>

# ============================================================================
#  PART 1: THE MODEL'S WEIGHTS
#     Every matrix is a plain double[] (row-major: M[i,j] = A[i*cols+j]).
#     Access weights by name: $P['Wq_2'] is the query matrix of layer index 2.
# ============================================================================
<#
  Weights are the model's trainable numbers. During training, we update
  them so the model assigns more probability to the actual continuation.
  During generation, the weights stay fixed.

  We arrange them in several tables. Tok stores a trainable sequence
  of numbers for each character. The layers process these sequences.

  Head is the weight matrix used in the final step. It takes the numbers
  produced by the layers and computes one score for each known character.
  After "alm", example scores might be:
    a: 3.2, e: 1.1, z: -0.8  (an illustration, not measured output).
  Softmax gives higher-scoring characters larger probabilities.
  We use those probabilities to select a character.
  This Head is different from the attention heads within the layers.

  The next two functions define the tables' sizes and starting values.
#>

$script:Config = @{
    VocabSize = 0
    BlockSize = $Block
    EmbedDim  = $Dim
    NumHeads  = $Heads
    NumLayers = $Layers
}

function Get-ParamShapes {
    <#
      Return the size of each weight table: name -> (rows, columns).
      With 65 characters and Dim=64, Tok holds 65 x 64 numbers.
      Each row belongs to one character.
      We use this size list to create and check the weights.
    #>
    param([hashtable]$Cfg)
    $d = $Cfg.EmbedDim; $ff = 4 * $d
    $shapes = [ordered]@{
        Tok = @($Cfg.VocabSize, $d)   # token embedding
        Pos = @($Cfg.BlockSize, $d)   # position embedding
    }
    for ($l = 0; $l -lt $Cfg.NumLayers; $l++) {
        $shapes["g1_$l"] = @(1, $d);   $shapes["b1_$l"] = @(1, $d)   # layernorm 1
        $shapes["Wq_$l"] = @($d, $d);  $shapes["Wk_$l"] = @($d, $d)
        $shapes["Wv_$l"] = @($d, $d);  $shapes["Wo_$l"] = @($d, $d)
        $shapes["g2_$l"] = @(1, $d);   $shapes["b2_$l"] = @(1, $d)   # layernorm 2
        $shapes["W1_$l"] = @($d, $ff); $shapes["W2_$l"] = @($ff, $d) # MLP
    }
    $shapes['gf'] = @(1, $d); $shapes['bf'] = @(1, $d)              # final layernorm
    $shapes['Head'] = @($d, $Cfg.VocabSize)
    return $shapes
}

function New-GptParams {
    <#
      Create the weight tables for an untrained model.
      Matrix entries start from a normal distribution with mean 0 and
      standard deviation 0.02. Different initial values allow units to
      learn different patterns.

      Exceptions: LayerNorm scales start at 1 and offsets at 0.
      For Wo and W2, which feed into residual additions, we also multiply
      the standard deviation by 1/sqrt(2*NumLayers).

      Seed sets the random number generator's starting state. With the same
      configuration in this environment, initialization can be repeated.
    #>
    param([hashtable]$Cfg, [int]$Seed = 1337)
    $rng = [Random]::new($Seed)
    $P = @{}
    $shapes = Get-ParamShapes $Cfg
    $resScale = 1.0 / [Math]::Sqrt(2.0 * $Cfg.NumLayers)
    foreach ($name in $shapes.Keys) {
        $n = $shapes[$name][0] * $shapes[$name][1]
        $a = [double[]]::new($n)
        if ($name -like 'g*') { for ($i = 0; $i -lt $n; $i++) { $a[$i] = 1.0 } }
        elseif ($name -like 'b*') { }
        else {
            $std = 0.02
            if ($name -like 'Wo_*' -or $name -like 'W2_*') { $std *= $resScale }
            for ($i = 0; $i -lt $n; $i++) {
                # Box-Muller: two uniform numbers -> one normally distributed number
                $u1 = 1.0 - $rng.NextDouble(); $u2 = $rng.NextDouble()
                $a[$i] = $std * [Math]::Sqrt(-2.0 * [Math]::Log($u1)) * [Math]::Cos(2.0 * [Math]::PI * $u2)
            }
        }
        $P[$name] = $a
    }
    return $P
}

# ============================================================================
#  PART 2: FORWARD + BACKWARD FOR ONE TEXT WINDOW
#
#     This is the heart of the script. A single, self-contained function, because
#     only this gets passed to the parallel threads (runspaces) as text. So it does NOT call
#     any other function; the matrix multiplications are local scriptblocks.
#
#     Input:   $Ids = T+1 character ids. Input = Ids[0..T-1], target = Ids[1..T]
#     Output:  @{ Loss = mean cross-entropy; Grads = name -> double[] }
#
#     Formulas: linear projections have no bias; LayerNorm has a trainable offset.
#       x0 = Tok[id] + Pos[t]
#       in every layer:
#         a  = LN1(x)                 q = a Wq, k = a Wk, v = a Wv
#         att = softmax(q k^T / sqrt(hd), causal)     o = att v
#         x  = x + o Wo
#         a2 = LN2(x)                 h = a2 W1,  gh = GELU(h)
#         x  = x + gh W2
#       af = LNf(x);  logits = af Head;  loss = -log softmax(logits)[target]
#
#     The backward pass is the same thing in reverse, following the chain rule.
# ============================================================================
<#
  A text window gives us input-target pairs. For example:

    text:      a l m a
    input:     a l m
    target:    l m a

  The three predictions are: a -> l, al -> m, alm -> a.
  The causal mask lets each position use itself and earlier positions.
  Even during training, it cannot read the answer from a later input.

  Forward: compute next-character probabilities at every position.
  The loss penalizes assigning little probability to the actual
  continuation. We average it over the positions.

  Backward: compute how the loss responds to small changes in each
  weight. This is the gradient. A small change delta changes the loss
  by approximately gradient * delta. We do not update weights here;
  Invoke-Train does that using the Adam optimizer.

  On a first read, follow FORWARD through the loss calculation.
  Return to BACKWARD after you understand generation.
  Matrix operations are defined locally because parallel tasks receive
  this function on its own, as text.
#>

function Compute-SeqGrad {
    param([hashtable]$P, [int[]]$Ids, [hashtable]$Cfg, [bool]$NeedGrad = $true)

    $d = [int]$Cfg.EmbedDim; $nL = [int]$Cfg.NumLayers; $nH = [int]$Cfg.NumHeads
    $hd = [int]($d / $nH); $ff = 4 * $d; $vocab = [int]$Cfg.VocabSize
    $seqLen = $Ids.Length - 1
    $attScale = 1.0 / [Math]::Sqrt([double]$hd)
    $geluC = [Math]::Sqrt(2.0 / [Math]::PI)

    # Matrix multiplication turns an input vector into a new vector.
    # Example: [2, 3] multiplied by the matrix with rows:
    #   [1, 4]
    #   [5, 6]
    # Result: [2*1 + 3*5, 2*4 + 3*6] = [17, 26].
    # Each output element is a weighted sum of the input.
    # MM multiplies; MMT multiplies by the transpose of the second matrix;
    # ATB adds a product using the transpose of the first matrix to G.
    # Transposing swaps row and column indices: B^T[i,j] = B[j,i].
    # Skipping zero input elements only avoids unnecessary multiplications.
    # C(ra x cb) = A(ra x ca) * B(ca x cb)
    $MM = {
        param([double[]]$A, [int]$ra, [int]$ca, [double[]]$B, [int]$cb)
        $C = [double[]]::new($ra * $cb)
        for ($i = 0; $i -lt $ra; $i++) {
            $ai = $i * $ca; $ci = $i * $cb
            for ($k = 0; $k -lt $ca; $k++) {
                $av = $A[$ai + $k]
                if ($av -eq 0.0) { continue }
                $bk = $k * $cb
                for ($j = 0; $j -lt $cb; $j++) { $C[$ci + $j] += $av * $B[$bk + $j] }
            }
        }
        return ,$C
    }
    # C(ra x rb) = A(ra x ca) * B(rb x ca)^T
    $MMT = {
        param([double[]]$A, [int]$ra, [int]$ca, [double[]]$B, [int]$rb)
        $C = [double[]]::new($ra * $rb)
        for ($i = 0; $i -lt $ra; $i++) {
            $ai = $i * $ca; $ci = $i * $rb
            for ($j = 0; $j -lt $rb; $j++) {
                $bj = $j * $ca; $s = 0.0
                for ($k = 0; $k -lt $ca; $k++) { $s += $A[$ai + $k] * $B[$bj + $k] }
                $C[$ci + $j] = $s
            }
        }
        return ,$C
    }
    # G(ca x cb) += A(ra x ca)^T * B(ra x cb)   (weight gradient)
    $ATB = {
        param([double[]]$A, [int]$ra, [int]$ca, [double[]]$B, [int]$cb, [double[]]$G)
        for ($i = 0; $i -lt $ra; $i++) {
            $ai = $i * $ca; $bi = $i * $cb
            for ($k = 0; $k -lt $ca; $k++) {
                $av = $A[$ai + $k]
                if ($av -eq 0.0) { continue }
                $gk = $k * $cb
                for ($j = 0; $j -lt $cb; $j++) { $G[$gk + $j] += $av * $B[$bi + $j] }
            }
        }
    }

    # ======================= FORWARD =======================
    # Use character id Ids[t] to select a row from Tok.
    # The row contains d trainable numbers; the id is only its row index.
    # Add the row for position t from Pos.
    # A two-dimensional example: [0.2, -0.1] + [0.0, 0.3] = [0.2, 0.2].
    # In the actual model, x stores d numbers for each position.
    $Tok = $P['Tok']; $Pos = $P['Pos']
    $x = [double[]]::new($seqLen * $d)
    for ($t = 0; $t -lt $seqLen; $t++) {
        $ti = $Ids[$t] * $d; $pi = $t * $d; $xi = $t * $d
        for ($j = 0; $j -lt $d; $j++) { $x[$xi + $j] = $Tok[$ti + $j] + $Pos[$pi + $j] }
    }

    $cache = [object[]]::new($nL)
    for ($l = 0; $l -lt $nL; $l++) {
        $g1 = $P["g1_$l"]; $b1 = $P["b1_$l"]; $Wq = $P["Wq_$l"]; $Wk = $P["Wk_$l"]
        $Wv = $P["Wv_$l"]; $Wo = $P["Wo_$l"]; $g2 = $P["g2_$l"]; $b2 = $P["b2_$l"]
        $W1 = $P["W1_$l"]; $W2 = $P["W2_$l"]
        $xIn = $x

        # --- LayerNorm 1 ---
        # At each position, subtract the mean of its d numbers, then divide
        # by sqrt(variance + 1e-5). The small constant prevents division by zero.
        # This makes the scale of different inputs more consistent.
        # The trainable g scales the normalized values and b shifts them.
        # There is no fixed lower or upper bound on the result.
        # We keep xhat and rstd for the backward calculation.
        $xhat1 = [double[]]::new($seqLen * $d); $rstd1 = [double[]]::new($seqLen)
        $a1 = [double[]]::new($seqLen * $d)
        for ($t = 0; $t -lt $seqLen; $t++) {
            $o1 = $t * $d; $mean = 0.0
            for ($j = 0; $j -lt $d; $j++) { $mean += $x[$o1 + $j] }
            $mean /= $d; $var = 0.0
            for ($j = 0; $j -lt $d; $j++) { $dev = $x[$o1 + $j] - $mean; $var += $dev * $dev }
            $rs = 1.0 / [Math]::Sqrt($var / $d + 1e-5); $rstd1[$t] = $rs
            for ($j = 0; $j -lt $d; $j++) {
                $xh = ($x[$o1 + $j] - $mean) * $rs
                $xhat1[$o1 + $j] = $xh
                $a1[$o1 + $j] = $xh * $g1[$j] + $b1[$j]
            }
        }

        # --- Q, K, V ---
        $q = & $MM $a1 $seqLen $d $Wq $d
        $k = & $MM $a1 $seqLen $d $Wk $d
        $v = & $MM $a1 $seqLen $d $Wv $d

        # --- Causal multi-head attention ---
        # att[h, t, u] = how much character t attends to character u (u <= t)
        #
        # Attention mixes information from other positions into the current one.
        # Three learned projections come from the same normalized input:
        #   q: used to compare the current position with the available positions;
        #   k: what q is compared against at each available position;
        #   v: the vectors we sum using the resulting weights.
        # The dot product of q and k gives a score, scaled by sqrt(hd).
        # Softmax turns these scores into nonnegative weights that sum to 1.
        # Example: weights [0.2, 0.3, 0.5] give 0.2*v0 + 0.3*v1 + 0.5*v2.
        # These are attention weights; next-character probabilities come later.
        # Position t can use positions 0..t, including itself.
        # Later positions are excluded because they would not exist during generation.
        # Each head computes its own weights on an hd-long slice of q, k, and v.
        # Heads can learn different patterns; their roles are not assigned in advance.
        # Before softmax, we subtract the largest score. This leaves the distribution
        # unchanged while avoiding exponentials of very large numbers.
        $att = [double[]]::new($nH * $seqLen * $seqLen)
        $o = [double[]]::new($seqLen * $d)
        for ($h = 0; $h -lt $nH; $h++) {
            $off = $h * $hd; $hBase = $h * $seqLen * $seqLen
            for ($t = 0; $t -lt $seqLen; $t++) {
                $qi = $t * $d + $off; $rowBase = $hBase + $t * $seqLen; $max = -1e300
                for ($u = 0; $u -le $t; $u++) {
                    $ki = $u * $d + $off; $s = 0.0
                    for ($j = 0; $j -lt $hd; $j++) { $s += $q[$qi + $j] * $k[$ki + $j] }
                    $s *= $attScale
                    $att[$rowBase + $u] = $s
                    if ($s -gt $max) { $max = $s }
                }
                $sum = 0.0
                for ($u = 0; $u -le $t; $u++) { $e = [Math]::Exp($att[$rowBase + $u] - $max); $att[$rowBase + $u] = $e; $sum += $e }
                for ($u = 0; $u -le $t; $u++) { $att[$rowBase + $u] /= $sum }
                # o[t] = sum_u att[t,u] * v[u]
                $oi = $t * $d + $off
                for ($u = 0; $u -le $t; $u++) {
                    $pw = $att[$rowBase + $u]; $vi = $u * $d + $off
                    for ($j = 0; $j -lt $hd; $j++) { $o[$oi + $j] += $pw * $v[$vi + $j] }
                }
            }
        }
        $proj = & $MM $o $seqLen $d $Wo $d
        $x1 = [double[]]::new($seqLen * $d)
        for ($i = 0; $i -lt $x1.Length; $i++) { $x1[$i] = $xIn[$i] + $proj[$i] }   # residual

        # --- LayerNorm 2 ---
        $xhat2 = [double[]]::new($seqLen * $d); $rstd2 = [double[]]::new($seqLen)
        $a2 = [double[]]::new($seqLen * $d)
        for ($t = 0; $t -lt $seqLen; $t++) {
            $o2 = $t * $d; $mean = 0.0
            for ($j = 0; $j -lt $d; $j++) { $mean += $x1[$o2 + $j] }
            $mean /= $d; $var = 0.0
            for ($j = 0; $j -lt $d; $j++) { $dev = $x1[$o2 + $j] - $mean; $var += $dev * $dev }
            $rs = 1.0 / [Math]::Sqrt($var / $d + 1e-5); $rstd2[$t] = $rs
            for ($j = 0; $j -lt $d; $j++) {
                $xh = ($x1[$o2 + $j] - $mean) * $rs
                $xhat2[$o2 + $j] = $xh
                $a2[$o2 + $j] = $xh * $g2[$j] + $b2[$j]
            }
        }

        # --- MLP: h = a2 W1 ; gh = GELU(h) ; m = gh W2 ---
        # The MLP processes each position separately, using the same weight matrices.
        # W1 expands the vector from d to 4*d, GELU applies a nonlinear function
        # element by element, and W2 maps the result back to d.
        # Without GELU, these two matrix products could be combined into one.
        # The full transformer also contains other nonlinear operations.
        # Here we compute GELU using the tanh approximation shown below.
        # The residual addition is x2 = x1 + m. The input reaches the next layer
        # directly, together with the contribution computed in m.
        $hpre = & $MM $a2 $seqLen $d $W1 $ff
        $gh = [double[]]::new($seqLen * $ff)
        for ($i = 0; $i -lt $gh.Length; $i++) {
            $u = $hpre[$i]
            $gh[$i] = 0.5 * $u * (1.0 + [Math]::Tanh($geluC * ($u + 0.044715 * $u * $u * $u)))
        }
        $m = & $MM $gh $seqLen $ff $W2 $d
        $x2 = [double[]]::new($seqLen * $d)
        for ($i = 0; $i -lt $x2.Length; $i++) { $x2[$i] = $x1[$i] + $m[$i] }     # residual

        $cache[$l] = @{ xhat1 = $xhat1; rstd1 = $rstd1; a1 = $a1; q = $q; k = $k; v = $v
                        att = $att; o = $o; xhat2 = $xhat2; rstd2 = $rstd2; a2 = $a2
                        hpre = $hpre; gh = $gh }
        $x = $x2
    }

    # --- Final LayerNorm + Head + softmax + loss (at EVERY position) ---
    # After the final LayerNorm, Head gives one score for each possible
    # character. These logits are not probabilities yet.
    # Softmax probabilities sum to 1 at each position.
    # The loss at a position is -ln(p), where p is the probability of the actual next character.
    # Examples: p=0.5 -> loss=0.693; p=0.1 -> 2.303; p=0.01 -> 4.605.
    # Giving the correct character more probability lowers this loss.
    # We average the loss over positions. A uniform distribution over
    # 65 characters gives ln(65), approximately 4.174.
    # This measures the current window. Generalization to new text should
    # be measured separately on text that was not used for training.
    $gf = $P['gf']; $bf = $P['bf']; $Head = $P['Head']
    $xhatF = [double[]]::new($seqLen * $d); $rstdF = [double[]]::new($seqLen)
    $af = [double[]]::new($seqLen * $d)
    for ($t = 0; $t -lt $seqLen; $t++) {
        $o3 = $t * $d; $mean = 0.0
        for ($j = 0; $j -lt $d; $j++) { $mean += $x[$o3 + $j] }
        $mean /= $d; $var = 0.0
        for ($j = 0; $j -lt $d; $j++) { $dev = $x[$o3 + $j] - $mean; $var += $dev * $dev }
        $rs = 1.0 / [Math]::Sqrt($var / $d + 1e-5); $rstdF[$t] = $rs
        for ($j = 0; $j -lt $d; $j++) {
            $xh = ($x[$o3 + $j] - $mean) * $rs
            $xhatF[$o3 + $j] = $xh
            $af[$o3 + $j] = $xh * $gf[$j] + $bf[$j]
        }
    }
    $logits = & $MM $af $seqLen $d $Head $vocab
    $probs = [double[]]::new($seqLen * $vocab)
    $loss = 0.0
    for ($t = 0; $t -lt $seqLen; $t++) {
        $li = $t * $vocab; $max = -1e300
        for ($c = 0; $c -lt $vocab; $c++) { if ($logits[$li + $c] -gt $max) { $max = $logits[$li + $c] } }
        $sum = 0.0
        for ($c = 0; $c -lt $vocab; $c++) { $e = [Math]::Exp($logits[$li + $c] - $max); $probs[$li + $c] = $e; $sum += $e }
        for ($c = 0; $c -lt $vocab; $c++) { $probs[$li + $c] /= $sum }
        $loss += -[Math]::Log([Math]::Max($probs[$li + $Ids[$t + 1]], 1e-300))
    }
    $loss /= $seqLen

    if (-not $NeedGrad) {
        $lastLogits = [double[]]::new($vocab)
        [Array]::Copy($logits, ($seqLen - 1) * $vocab, $lastLogits, 0, $vocab)
        return @{ Loss = $loss; LastLogits = $lastLogits }
    }

    # ======================= BACKWARD =======================
    # dlogits, daf, and dx are derivatives of the loss with respect to
    # intermediate results. A positive value means that a small increase
    # in that element would locally increase the loss; a negative value
    # means it would decrease it. The standalone $d still means model dimension.
    # $G collects gradients for the weight tables, using the same keys as $P.
    $G = @{}

    # The derivative of softmax + cross-entropy with respect to logits is (p - target) / T.
    # The target vector is 1 at the correct character and 0 elsewhere.
    # Example: p=[0.2, 0.5, 0.3], with the second character as the target.
    # Then p-target=[0.2, -0.5, 0.3]. Divide this by T because
    # the loss was averaged over T positions.
    $dlogits = $probs
    for ($t = 0; $t -lt $seqLen; $t++) {
        $li = $t * $vocab
        $dlogits[$li + $Ids[$t + 1]] -= 1.0
        for ($c = 0; $c -lt $vocab; $c++) { $dlogits[$li + $c] /= $seqLen }
    }
    # Head: logits = af Head  ->  dHead = af^T dlogits ; daf = dlogits Head^T
    $dHead = [double[]]::new($d * $vocab)
    & $ATB $af $seqLen $d $dlogits $vocab $dHead
    $G['Head'] = $dHead
    $daf = & $MMT $dlogits $seqLen $vocab $Head $d

    # Changing one input element also changes the mean and variance,
    # affecting the other normalized elements at the same position.
    # The two mean terms in the backward formula account for this shared dependence.
    # We use the same formula later for LN2 and LN1.
    # LayerNorm backward (shared formula):
    #   dg += dy*xhat ; db += dy ; dxhat = dy*g
    #   dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat*xhat))
    $dgf = [double[]]::new($d); $dbf = [double[]]::new($d)
    $dx = [double[]]::new($seqLen * $d)
    for ($t = 0; $t -lt $seqLen; $t++) {
        $o4 = $t * $d; $m1 = 0.0; $m2 = 0.0
        for ($j = 0; $j -lt $d; $j++) {
            $dy = $daf[$o4 + $j]; $xh = $xhatF[$o4 + $j]
            $dgf[$j] += $dy * $xh; $dbf[$j] += $dy
            $dxh = $dy * $gf[$j]; $m1 += $dxh; $m2 += $dxh * $xh
        }
        $m1 /= $d; $m2 /= $d; $rs = $rstdF[$t]
        for ($j = 0; $j -lt $d; $j++) {
            $dxh = $daf[$o4 + $j] * $gf[$j]
            $dx[$o4 + $j] = $rs * ($dxh - $m1 - $xhatF[$o4 + $j] * $m2)
        }
    }
    $G['gf'] = $dgf; $G['bf'] = $dbf

    for ($l = $nL - 1; $l -ge 0; $l--) {
        $cc = $cache[$l]
        $g1 = $P["g1_$l"]; $Wq = $P["Wq_$l"]; $Wk = $P["Wk_$l"]; $Wv = $P["Wv_$l"]
        $Wo = $P["Wo_$l"]; $g2 = $P["g2_$l"]; $W1 = $P["W1_$l"]; $W2 = $P["W2_$l"]
        $xhat1 = $cc.xhat1; $rstd1 = $cc.rstd1; $a1 = $cc.a1; $q = $cc.q; $k = $cc.k; $v = $cc.v
        $att = $cc.att; $o = $cc.o; $xhat2 = $cc.xhat2; $rstd2 = $cc.rstd2; $a2 = $cc.a2
        $hpre = $cc.hpre; $gh = $cc.gh

        # --- MLP backward: x2 = x1 + gh W2 ---
        $dW2 = [double[]]::new($ff * $d)
        & $ATB $gh $seqLen $ff $dx $d $dW2
        $dgh = & $MMT $dx $seqLen $d $W2 $ff
        # GELU derivative
        for ($i = 0; $i -lt $dgh.Length; $i++) {
            $u = $hpre[$i]
            $z = $geluC * ($u + 0.044715 * $u * $u * $u)
            $th = [Math]::Tanh($z)
            $dgh[$i] *= 0.5 * (1.0 + $th) + 0.5 * $u * (1.0 - $th * $th) * $geluC * (1.0 + 3.0 * 0.044715 * $u * $u)
        }
        $dW1 = [double[]]::new($d * $ff)
        & $ATB $a2 $seqLen $d $dgh $ff $dW1
        $da2 = & $MMT $dgh $seqLen $ff $W1 $d
        $G["W1_$l"] = $dW1; $G["W2_$l"] = $dW2

        # LN2 backward, residual: dx1 = dx + LNback(da2)
        $dg2 = [double[]]::new($d); $db2 = [double[]]::new($d)
        $dx1 = [double[]]::new($seqLen * $d)
        for ($t = 0; $t -lt $seqLen; $t++) {
            $o5 = $t * $d; $m1 = 0.0; $m2 = 0.0
            for ($j = 0; $j -lt $d; $j++) {
                $dy = $da2[$o5 + $j]; $xh = $xhat2[$o5 + $j]
                $dg2[$j] += $dy * $xh; $db2[$j] += $dy
                $dxh = $dy * $g2[$j]; $m1 += $dxh; $m2 += $dxh * $xh
            }
            $m1 /= $d; $m2 /= $d; $rs = $rstd2[$t]
            for ($j = 0; $j -lt $d; $j++) {
                $dxh = $da2[$o5 + $j] * $g2[$j]
                $dx1[$o5 + $j] = $dx[$o5 + $j] + $rs * ($dxh - $m1 - $xhat2[$o5 + $j] * $m2)
            }
        }
        $G["g2_$l"] = $dg2; $G["b2_$l"] = $db2

        # --- Attention backward: x1 = xIn + o Wo ---
        # The result o is a weighted sum of the v vectors. Going backward,
        # we compute gradients for both v and the attention weights.
        # The contribution to v is multiplied by the corresponding attention weight.
        # Gradients for attention weights pass through softmax to q and k.
        # We also collect gradients for the Wq, Wk, and Wv weight matrices.
        $dWo = [double[]]::new($d * $d)
        & $ATB $o $seqLen $d $dx1 $d $dWo
        $do = & $MMT $dx1 $seqLen $d $Wo $d
        $G["Wo_$l"] = $dWo

        $dq = [double[]]::new($seqLen * $d); $dk = [double[]]::new($seqLen * $d); $dv = [double[]]::new($seqLen * $d)
        for ($h = 0; $h -lt $nH; $h++) {
            $off = $h * $hd; $hBase = $h * $seqLen * $seqLen
            for ($t = 0; $t -lt $seqLen; $t++) {
                $rowBase = $hBase + $t * $seqLen; $oi = $t * $d + $off
                # dv[u] += att[t,u] * do[t] ;  datt[t,u] = do[t] . v[u]
                $datt = [double[]]::new($t + 1); $dot = 0.0
                for ($u = 0; $u -le $t; $u++) {
                    $pw = $att[$rowBase + $u]; $vi = $u * $d + $off; $s = 0.0
                    for ($j = 0; $j -lt $hd; $j++) {
                        $dv[$vi + $j] += $pw * $do[$oi + $j]
                        $s += $do[$oi + $j] * $v[$vi + $j]
                    }
                    $datt[$u] = $s; $dot += $pw * $s
                }
                # softmax backward: dscore = att * (datt - sum(att*datt))
                $qi = $t * $d + $off
                for ($u = 0; $u -le $t; $u++) {
                    $ds = $att[$rowBase + $u] * ($datt[$u] - $dot) * $attScale
                    $ki = $u * $d + $off
                    for ($j = 0; $j -lt $hd; $j++) {
                        $dq[$qi + $j] += $ds * $k[$ki + $j]
                        $dk[$ki + $j] += $ds * $q[$qi + $j]
                    }
                }
            }
        }
        $dWq = [double[]]::new($d * $d); $dWk = [double[]]::new($d * $d); $dWv = [double[]]::new($d * $d)
        & $ATB $a1 $seqLen $d $dq $d $dWq
        & $ATB $a1 $seqLen $d $dk $d $dWk
        & $ATB $a1 $seqLen $d $dv $d $dWv
        $G["Wq_$l"] = $dWq; $G["Wk_$l"] = $dWk; $G["Wv_$l"] = $dWv
        $da1 = & $MMT $dq $seqLen $d $Wq $d
        $tmpK = & $MMT $dk $seqLen $d $Wk $d
        $tmpV = & $MMT $dv $seqLen $d $Wv $d
        for ($i = 0; $i -lt $da1.Length; $i++) { $da1[$i] += $tmpK[$i] + $tmpV[$i] }

        # LN1 backward, residual: dx0 = dx1 + LNback(da1)
        $dg1 = [double[]]::new($d); $db1 = [double[]]::new($d)
        $dx0 = [double[]]::new($seqLen * $d)
        for ($t = 0; $t -lt $seqLen; $t++) {
            $o6 = $t * $d; $m1 = 0.0; $m2 = 0.0
            for ($j = 0; $j -lt $d; $j++) {
                $dy = $da1[$o6 + $j]; $xh = $xhat1[$o6 + $j]
                $dg1[$j] += $dy * $xh; $db1[$j] += $dy
                $dxh = $dy * $g1[$j]; $m1 += $dxh; $m2 += $dxh * $xh
            }
            $m1 /= $d; $m2 /= $d; $rs = $rstd1[$t]
            for ($j = 0; $j -lt $d; $j++) {
                $dxh = $da1[$o6 + $j] * $g1[$j]
                $dx0[$o6 + $j] = $dx1[$o6 + $j] + $rs * ($dxh - $m1 - $xhat1[$o6 + $j] * $m2)
            }
        }
        $G["g1_$l"] = $dg1; $G["b1_$l"] = $db1
        $dx = $dx0
    }

    # --- Embedding gradient ---
    # A character can occur at several positions. Each occurrence used
    # the same row of Tok, so we add its gradient contribution to that row.
    # This is why the update uses += rather than assignment.
    # Contributions can reinforce or partially cancel each other.
    # Position-vector gradients go into the corresponding rows of Pos.
    $dTok = [double[]]::new($vocab * $d); $dPos = [double[]]::new([int]$Cfg.BlockSize * $d)
    for ($t = 0; $t -lt $seqLen; $t++) {
        $ti = $Ids[$t] * $d; $xi = $t * $d
        for ($j = 0; $j -lt $d; $j++) { $dTok[$ti + $j] += $dx[$xi + $j]; $dPos[$xi + $j] += $dx[$xi + $j] }
    }
    $G['Tok'] = $dTok; $G['Pos'] = $dPos

    return @{ Loss = $loss; Grads = $G }
}

# ============================================================================
#  PART 3: SAVE / LOAD
# ============================================================================
<#
  A checkpoint is a saved training state.
  The JSON contains model dimensions, character ids, weights,
  Adam's M and V arrays, and the step count. We load it to resume
  training or to generate text.

  The random number generator state is not saved, so resuming may
  select different training windows than an uninterrupted run.
  The number of additional steps requested also affects the remaining
  learning-rate schedule.
#>

function Save-Checkpoint {
    # Save the weights, Adam state, and step count as JSON.
    # Write the .tmp file first, then replace the destination with it.
    # The previous checkpoint stays at the destination while JSON is being written.
    param([string]$Path, [hashtable]$P, [hashtable]$Adam, [int]$Step)
    $out = [ordered]@{
        Version  = 2
        Config   = $script:Config
        Step     = $Step
        # only the Vocab list (id -> character); we don't save CharToId, because the JSON reader
        # does not tolerate keys that differ only in upper/lower case
        Vocab    = @(0..($script:Config.VocabSize - 1) | ForEach-Object { $script:IdToChar[$_] })
        Params   = $P
        AdamM    = $Adam.M
        AdamV    = $Adam.V
    }
    $tmp = "$Path.tmp"
    $out | ConvertTo-Json -Depth 5 -Compress | Set-Content -Path $tmp -Encoding UTF8
    Move-Item -Path $tmp -Destination $Path -Force
}

function Load-Checkpoint {
    # Input: the path of the JSON file. Output: @{ Params; Adam; Step } - the mirror image
    # of Save-Checkpoint. As a side effect it sets the global Config and the character dictionary
    # (CharToId / IdToChar), because without these the loaded weights are unusable.
    # It rejects the old checkpoint format, which only contained the Head layer.
    param([string]$Path)
    $json = Get-Content -Path $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not ($json.PSObject.Properties.Name -contains 'Version') -or $json.Version -ne 2) {
        if ($script:Lang -eq 'en') {
            throw "Unsupported weight-file version: $Path. This script requires a version 2 checkpoint."
        } else {
            throw "Nem támogatott súlyfájlverzió: $Path. Ehhez a scripthez 2-es verziójú checkpoint kell."
        }
    }
    foreach ($key in 'VocabSize', 'BlockSize', 'EmbedDim', 'NumHeads', 'NumLayers') { $script:Config[$key] = [int]$json.Config.$key }
    # IMPORTANT: PowerShell hashtable keys are case-insensitive ('a' == 'A'),
    # so the character->id table is a case-sensitive Dictionary.
    $script:CharToId = [System.Collections.Generic.Dictionary[string,int]]::new([StringComparer]::Ordinal); $script:IdToChar = @{}
    if ($json.PSObject.Properties.Name -contains 'Vocab') {
        $vocabList = @($json.Vocab)
        for ($i = 0; $i -lt $vocabList.Count; $i++) { $script:CharToId[[string]$vocabList[$i]] = $i; $script:IdToChar[$i] = [string]$vocabList[$i] }
    } else {
        # Old checkpoint (trained with the case-merging bug): the model learned lowercase text,
        # so the output is lowercase too, and we lowercase the letters of the prompt.
        foreach ($prop in $json.CharToId.PSObject.Properties) { $ch = ([string]$prop.Name).ToLowerInvariant(); $script:CharToId[$ch] = [int]$prop.Value; $script:CharToId[$ch.ToUpperInvariant()] = [int]$prop.Value; $script:IdToChar[[int]$prop.Value] = $ch }
        Write-Host $(if ($script:Lang -eq 'en') { 'Legacy character mapping: uppercase and lowercase letters share ids, so output will be lowercase.' } else { 'Régi karaktertábla: a kis- és nagybetűk közös azonosítót használnak, ezért a kimenet kisbetűs lesz.' }) -ForegroundColor Yellow
    }
    $P = @{}; $M = @{}; $V = @{}
    foreach ($prop in $json.Params.PSObject.Properties) { $P[$prop.Name] = [double[]]$prop.Value }
    foreach ($prop in $json.AdamM.PSObject.Properties) { $M[$prop.Name] = [double[]]$prop.Value }
    foreach ($prop in $json.AdamV.PSObject.Properties) { $V[$prop.Name] = [double[]]$prop.Value }
    return @{ Params = $P; Adam = @{ M = $M; V = $V }; Step = [int]$json.Step }
}

# ============================================================================
#  PART 4: TRAINING (Adam, parallel batch)
# ============================================================================
<#
  One training step:
    1. Select BatchSize fragments from the training text.
       Each has BlockSize+1 characters: BlockSize inputs and shifted targets.
    2. Compute loss and gradients for each window.
       Windows independently read the same weights.
    3. Average the gradients. If their combined norm exceeds 1,
       scale them down proportionally.
    4. Adam updates weights using moving averages of the gradients.
    5. Print measurements and periodically save a checkpoint.

  A batch is the group of windows used for one weight update.
  Threads limits parallelism; it does not set the batch size.
  The printed loss was measured on the training windows before the update.
#>

function Invoke-Train {
    # Input: the corpus file and the number of steps to run. Output: no
    # return value; the weight file (checkpoint) is updated on disk, and
    # progress goes to the console. If a weight file already exists, it resumes from there.
    param([string]$Corpus, [int]$Steps)
    Write-Host $(if ($script:Lang -eq 'en') { "`n=== Training: updating weights in every layer ===" } else { "`n=== Tanítás: minden réteg súlyai frissülnek ===" }) -ForegroundColor Yellow
    $text = [System.IO.File]::ReadAllText($Corpus)

    $step0 = 0
    if (Test-Path $WeightsFile) {
        Write-Host $(if ($script:Lang -eq 'en') { "Resuming training from: $WeightsFile" } else { "Tanítás folytatása ebből a mentésből: $WeightsFile" }) -ForegroundColor Yellow
        $ck = Load-Checkpoint $WeightsFile
        $P = $ck.Params; $adam = $ck.Adam; $step0 = $ck.Step
        # GPU export (train_gpu.py): no Adam state, starts from zero
        foreach ($name in $P.Keys) {
            if (-not $adam.M.ContainsKey($name)) { $adam.M[$name] = [double[]]::new($P[$name].Length); $adam.V[$name] = [double[]]::new($P[$name].Length) }
        }
    } else {
        $chars = [char[]]([System.Linq.Enumerable]::ToArray([System.Linq.Enumerable]::OrderBy([System.Linq.Enumerable]::Distinct([char[]]$text.ToCharArray()), [Func[char,int]]{ param($c) [int]$c })))
        $script:CharToId = [System.Collections.Generic.Dictionary[string,int]]::new([StringComparer]::Ordinal); $script:IdToChar = @{}
        for ($i = 0; $i -lt $chars.Count; $i++) { $script:CharToId[[string]$chars[$i]] = $i; $script:IdToChar[$i] = [string]$chars[$i] }
        $script:Config.VocabSize = $chars.Count
        $P = New-GptParams $script:Config
        $adam = @{ M = @{}; V = @{} }
        foreach ($name in $P.Keys) { $adam.M[$name] = [double[]]::new($P[$name].Length); $adam.V[$name] = [double[]]::new($P[$name].Length) }
    }
    $cfg = $script:Config
    $nParams = 0; foreach ($name in $P.Keys) { $nParams += $P[$name].Length }
    $modelFmt = if ($script:Lang -eq 'en') { "Model: {0} layers; {1} dimensions; {2} heads; {3}-character window; {4} known characters; {5:N0} parameters" } else { "Modell: {0} réteg; {1} dimenzió; {2} fej; {3} karakteres ablak; {4} ismert karakter; {5:N0} paraméter" }
    Write-Host ($modelFmt -f $cfg.NumLayers, $cfg.EmbedDim, $cfg.NumHeads, $cfg.BlockSize, $cfg.VocabSize, $nParams) -ForegroundColor Cyan
    $batchFmt = if ($script:Lang -eq 'en') { "{0} windows per update; up to {1} parallel tasks; learning rate {2}; {3} additional steps; completed so far: {4}" } else { "Frissítésenként {0} ablak; legfeljebb {1} párhuzamos feladat; tanulási ráta {2}; további {3} lépés; eddig kész: {4}" }
    Write-Host ($batchFmt -f $BatchSize, $Threads, $LearningRate, $Steps, $step0) -ForegroundColor Cyan
    if ($script:Lang -eq 'en') {
        Write-Host 'loss: mean loss on this batch; avg20: mean of up to 20 recent batches.'
        Write-Host 'gnorm: gradient norm before clipping; lr: current learning rate; ETA: estimated time remaining.'
    } else {
        Write-Host 'loss: az aktuális batch átlagos vesztesége; avg20: legfeljebb 20 legutóbbi batch átlaga.'
        Write-Host 'gnorm: gradiensnorma a korlátozás előtt; lr: aktuális tanulási ráta; ETA: becsült hátralévő idő.'
    }

    # The per-window forward+backward function as text, so the threads can receive it
    $fnCode = ${function:Compute-SeqGrad}.ToString()
    $rng = [Random]::new()
    $block = $cfg.BlockSize
    $beta1 = 0.9; $beta2 = 0.99; $eps = 1e-8
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lossWindow = [System.Collections.Generic.Queue[double]]::new()
    $paramNames = @($P.Keys)

    for ($step = $step0 + 1; $step -le $step0 + $Steps; $step++) {
        # --- batch: BatchSize random windows ---
        # We cut block+1 characters out of the text at a random spot: the first block is the
        # input, the block shifted by one is the "correct answer" (always the next letter).
        $batch = [object[]]::new($BatchSize)
        for ($b = 0; $b -lt $BatchSize; $b++) {
            $start = $rng.Next(0, $text.Length - $block - 1)
            $ids = [int[]]::new($block + 1)
            for ($i = 0; $i -le $block; $i++) { $ids[$i] = $script:CharToId[[string]$text[$start + $i]] }
            $batch[$b] = $ids
        }

        # --- forward+backward in parallel (the weights are passed by reference) ---
        # Every window runs on its own thread, because they are independent: each one only
        # READS the weights and returns its own gradient. The threads do not write
        # shared data, so no locking is needed. $using: is the PowerShell way to
        # reach outer variables from inside the thread.
        if ($Threads -gt 1) {
            $results = $batch | ForEach-Object -ThrottleLimit $Threads -Parallel {
                $f = [scriptblock]::Create($using:fnCode)
                & $f $using:P $_ $using:cfg $true
            }
        } else {
            $results = foreach ($ids in $batch) { Compute-SeqGrad $P $ids $cfg $true }
        }

        # --- averaging the gradients + global norm (clip 1.0) ---
        # We add up the gradients of the 8 windows and divide by 8. The "clip" is a safety
        # brake: if the total length (norm) of the gradient is above 1, we scale it down proportionally.
        # This way an odd text snippet cannot cause a huge, damaging step.
        $loss = 0.0
        $gsum = @{}
        foreach ($name in $paramNames) { $gsum[$name] = [double[]]::new($P[$name].Length) }
        foreach ($r in $results) {
            $loss += $r.Loss
            foreach ($name in $paramNames) {
                $src = $r.Grads[$name]; $dst = $gsum[$name]
                for ($i = 0; $i -lt $dst.Length; $i++) { $dst[$i] += $src[$i] }
            }
        }
        $loss /= $BatchSize
        $norm2 = 0.0
        foreach ($name in $paramNames) {
            $g = $gsum[$name]
            for ($i = 0; $i -lt $g.Length; $i++) { $g[$i] /= $BatchSize; $norm2 += $g[$i] * $g[$i] }
        }
        $clip = 1.0; $gnorm = [Math]::Sqrt($norm2)
        if ($gnorm -gt 1.0) { $clip = 1.0 / $gnorm }

        # --- Adam ---
        # A basic gradient step would subtract lr * gradient from each weight.
        # Adam keeps two exponential moving averages per weight:
        #   m: average gradient, incorporating previous directions;
        #   v: average squared gradient, used to scale the update.
        # v does not measure the model's uncertainty.
        # bc1 and bc2 correct the bias from initializing the averages at zero.
        # The update uses corrected m / (sqrt(corrected v) + eps), multiplied
        # by the current learning rate lr.
        # During steps 1..100, lr increases gradually. After that it follows
        # a cosine curve toward 10% of LearningRate at the planned end of the run.
        # A new 20-step demonstration stays within the warmup phase throughout.
        $total = $step0 + $Steps
        if ($step -le 100) { $lr = $LearningRate * $step / 100.0 }
        else { $prog = ($step - 100) / [Math]::Max(1.0, $total - 100); $lr = $LearningRate * (0.1 + 0.9 * 0.5 * (1 + [Math]::Cos([Math]::PI * $prog))) }
        $bc1 = 1.0 - [Math]::Pow($beta1, $step); $bc2 = 1.0 - [Math]::Pow($beta2, $step)
        foreach ($name in $paramNames) {
            $pa = $P[$name]; $g = $gsum[$name]; $m = $adam.M[$name]; $v = $adam.V[$name]
            for ($i = 0; $i -lt $pa.Length; $i++) {
                $gi = $g[$i] * $clip
                $m[$i] = $beta1 * $m[$i] + (1 - $beta1) * $gi
                $v[$i] = $beta2 * $v[$i] + (1 - $beta2) * $gi * $gi
                $pa[$i] -= $lr * ($m[$i] / $bc1) / ([Math]::Sqrt($v[$i] / $bc2) + $eps)
            }
        }

        $lossWindow.Enqueue($loss); if ($lossWindow.Count -gt 20) { [void]$lossWindow.Dequeue() }
        $done = $step - $step0
        if ($step % 10 -eq 0 -or $done -eq 1) {
            $avg = 0.0; foreach ($l in $lossWindow) { $avg += $l }; $avg /= $lossWindow.Count
            $perStep = $sw.Elapsed.TotalSeconds / $done
            $eta = [TimeSpan]::FromSeconds($perStep * ($step0 + $Steps - $step))
            $stepFmt = if ($script:Lang -eq 'en') { "  step {0,6}/{1}  loss={2:F3} (avg20={3:F3})  gnorm={4:F2}  lr={5:E1}  {6:F1}s/step  ETA {7:d\.hh\:mm\:ss}" } else { "  lépés {0,6}/{1}  loss={2:F3} (avg20={3:F3})  gnorm={4:F2}  lr={5:E1}  {6:F1}s/lépés  ETA {7:d\.hh\:mm\:ss}" }
            Write-Host ($stepFmt -f $step, $total, $loss, $avg, $gnorm, $lr, $perStep, $eta) -ForegroundColor DarkGray
        }
        if ($step % $CheckpointEvery -eq 0) {
            Save-Checkpoint $WeightsFile $P $adam $step
            $saveFmt = if ($script:Lang -eq 'en') { "  Checkpoint saved at step {0}; elapsed: {1:hh\:mm\:ss}" } else { "  Mentés kész: {0}. lépés; eltelt idő: {1:hh\:mm\:ss}" }
            Write-Host ($saveFmt -f $step, $sw.Elapsed) -ForegroundColor Green
        }
    }
    Save-Checkpoint $WeightsFile $P $adam ($step0 + $Steps)
    Write-Host $(if ($script:Lang -eq 'en') { "Training finished. Weight file: $WeightsFile" } else { "A tanítás befejeződött. Súlyfájl: $WeightsFile" }) -ForegroundColor Green
}

# ============================================================================
#  PART 5: GRADIENT CHECK
#     Check gradients on a small model using a numerical estimate.
#     Estimate: (L(w+eps) - L(w-eps)) / (2*eps).
#     Compare this with the backward result at the sampled points.
# ============================================================================
<#
  A sign or indexing mistake can make backward return an incorrect gradient.
  We check it by another route: slightly increase a selected weight,
  then decrease it, and recompute the loss in each case.
  Their difference estimates the derivative with respect to that weight.

  We check four selected elements per parameter group on a fixed small
  model. The numerical result is approximate: it depends on eps and
  floating-point rounding. Agreement supports backward correctness at
  these points, but is not a proof for every possible input.

  The check reports agreement if all measured relative errors are below 1e-4.
  A gradient check alone does not measure how well the model learns.
#>

function Invoke-GradCheck {
    # Input: none (it builds itself a tiny, fixed model). Output: to the console,
    # the largest relative difference per weight group, and finally GRADIENT OK / FAILED.
    # It measures 4 random weights from every group - a full check would be too slow.
    $script:Config = @{ VocabSize = 11; BlockSize = 6; EmbedDim = 8; NumHeads = 2; NumLayers = 2 }
    $cfg = $script:Config
    $P = New-GptParams $cfg -Seed 7
    # larger weights, so the non-linearities "come alive" too
    $rng = [Random]::new(3)
    foreach ($name in @($P.Keys)) { if ($name -notlike 'g*' -and $name -notlike 'b*') { $a = $P[$name]; for ($i = 0; $i -lt $a.Length; $i++) { $a[$i] = ($rng.NextDouble() * 2 - 1) * 0.5 } } else { $a = $P[$name]; for ($i = 0; $i -lt $a.Length; $i++) { $a[$i] += ($rng.NextDouble() * 2 - 1) * 0.3 } } }
    $ids = [int[]]@(3, 7, 1, 9, 0, 5, 2)
    $res = Compute-SeqGrad $P $ids $cfg $true
    Write-Host ("Loss = {0:F6}" -f $res.Loss)
    $eps = 1e-5; $worst = 0.0
    foreach ($name in (Get-ParamShapes $cfg).Keys) {
        $a = $P[$name]; $g = $res.Grads[$name]
        $maxRel = 0.0
        for ($trial = 0; $trial -lt 4; $trial++) {
            $i = $rng.Next(0, $a.Length)
            $orig = $a[$i]
            $a[$i] = $orig + $eps; $lp = (Compute-SeqGrad $P $ids $cfg $false).Loss
            $a[$i] = $orig - $eps; $lm = (Compute-SeqGrad $P $ids $cfg $false).Loss
            $a[$i] = $orig
            $num = ($lp - $lm) / (2 * $eps)
            $rel = [Math]::Abs($num - $g[$i]) / [Math]::Max(1e-8, [Math]::Abs($num) + [Math]::Abs($g[$i]))
            if ($rel -gt $maxRel) { $maxRel = $rel }
        }
        if ($maxRel -gt $worst) { $worst = $maxRel }
        $color = if ($maxRel -lt 1e-4) { 'Green' } else { 'Red' }
        $relFmt = if ($script:Lang -eq 'en') { "  {0,-8} largest measured relative difference = {1:E2}" } else { "  {0,-8} legnagyobb mért relatív eltérés = {1:E2}" }
        Write-Host ($relFmt -f $name, $maxRel) -ForegroundColor $color
    }
    if ($worst -lt 1e-4) {
        Write-Host $(if ($script:Lang -eq 'en') { "The checked gradients agree with the numerical estimate." } else { "A vizsgált gradiensek egyeznek a numerikus közelítéssel." }) -ForegroundColor Green
    } else {
        Write-Host $(if ($script:Lang -eq 'en') { "At least one checked gradient differs by 1e-4 or more." } else { "Legalább egy vizsgált gradiens eltérése eléri az 1e-4 határt." }) -ForegroundColor Red
    }
}

# ============================================================================
#  PART 6: GENERATION
# ============================================================================
<#
  During generation, the weights are fixed. The loop is:
    1. Process the prompt characters to obtain next-character scores.
    2. Convert scores to probabilities using the temperature setting.
    3. Select and print one character.
    4. Process that character with the model, then repeat.

  Temperature is positive. Below 1, larger probabilities dominate;
  above 1, the distribution becomes more uniform. 1 gives ordinary softmax.
  With top-k filtering, only the K most likely characters remain eligible.
  TopK=0 disables filtering. TopK=1 always chooses a character with
  the highest score, so sampling adds no variety.

  The KV-cache retains previously computed key and value vectors.
  The MathNet version speeds up the same model calculations.
#>

function Sample-FromProbs {
    # Select a character according to the supplied probabilities.
    # For [0.2, 0.3, 0.5], the three characters have chances of 20%, 30%, and 50%.
    # A random number selects an interval of the cumulative distribution.
    # With top-k, keep the K largest probabilities and sample from their sum.
    # This is equivalent to renormalizing the probabilities of the retained characters.
    param([double[]]$Probs, [int]$TopK = 0)
    $rng = [Random]::new()
    if ($TopK -gt 0 -and $TopK -lt $Probs.Length) {
        $indexed = 0..($Probs.Length - 1) | Sort-Object { $Probs[$_] } -Descending | Select-Object -First $TopK
        $sum = 0.0; foreach ($ix in $indexed) { $sum += $Probs[$ix] }
        $r = $rng.NextDouble() * $sum; $c = 0.0
        foreach ($ix in $indexed) { $c += $Probs[$ix]; if ($r -le $c) { return $ix } }
        return $indexed[-1]
    }
    $r = $rng.NextDouble(); $c = 0.0
    for ($i = 0; $i -lt $Probs.Length; $i++) { $c += $Probs[$i]; if ($r -le $c) { return $i } }
    return $Probs.Length - 1
}

# The KV-cache keeps earlier positions' k and v vectors for each layer.
# With fixed weights and unchanged earlier positions, these vectors do not
# need to be recomputed for every new character. The new q still has to be
# compared with stored k vectors, and the result still uses a sum of stored v vectors.
# The speedup depends on model size and window length.
# When the cache fills, the code retains approximately the last quarter
# of the window and reprocesses it with positions numbered from 0.
# Text from the earlier part of the window leaves the available context.
# The cache is runtime state; clearing it does not change the weights.
# ============================================================================
#  OPTIONAL ACCELERATION: MathNet.Numerics
#  Uses lib/MathNet.Numerics.dll for matrix operations.
#  It applies the same weights and model steps as the PowerShell version.
#  If the library cannot be loaded, or -NoFast is set, the PowerShell version runs.
#  Floating-point results may differ slightly.
#  On a first read, follow Forward-Token.
# ============================================================================

$script:Fast = $false
$script:FastW = $null

function Initialize-Fast {
    # Input: the loaded weights. Output: none; it sets the $script:Fast switch
    # and $script:FastW (the weight matrices "wrapped" into MathNet, without copying).
    # If the DLL is not there or -NoFast is given, it silently falls back to the pure PowerShell path.
    param([hashtable]$P)
    $script:Fast = $false
    if ($NoFast) { return }
    $dll = Join-Path $PSScriptRoot 'lib/MathNet.Numerics.dll'
    if (-not (Test-Path $dll)) { return }
    try { Add-Type -Path $dll -ErrorAction Stop } catch { return }
    $cfg = $script:Config; $d = [int]$cfg.EmbedDim; $ff = 4 * $d; $vocab = [int]$cfg.VocabSize
    $DM = [MathNet.Numerics.LinearAlgebra.Double.DenseMatrix]
    # The row-major W (in x out) is exactly the column-major storage of W^T (out x in):
    # it becomes a matrix without copying, and y = W^T-matrix * a  ==  a * W.
    $FW = @{}
    for ($l = 0; $l -lt $cfg.NumLayers; $l++) {
        foreach ($nm in 'Wq', 'Wk', 'Wv', 'Wo') { $FW["${nm}_$l"] = $DM::OfColumnMajor($d, $d, $P["${nm}_$l"]) }
        $FW["W1_$l"] = $DM::OfColumnMajor($ff, $d, $P["W1_$l"])
        $FW["W2_$l"] = $DM::OfColumnMajor($d, $ff, $P["W2_$l"])
    }
    $FW['Head'] = $DM::OfColumnMajor($vocab, $d, $P['Head'])
    $script:FastW = $FW
    $script:Fast = $true
}

function New-FastCache {
    <# KV-cache for the fast path: per head one (block x hd) matrix for K and for V, plus work buffers. #>
    $cfg = $script:Config; $d = [int]$cfg.EmbedDim; $nL = [int]$cfg.NumLayers; $nH = [int]$cfg.NumHeads
    $hd = [int]($d / $nH); $ff = 4 * $d; $vocab = [int]$cfg.VocabSize; $block = [int]$cfg.BlockSize
    $DM = [MathNet.Numerics.LinearAlgebra.Double.DenseMatrix]; $DV = [MathNet.Numerics.LinearAlgebra.Double.DenseVector]
    $Kc = [object[]]::new($nL * $nH); $Vc = [object[]]::new($nL * $nH)
    for ($i = 0; $i -lt $Kc.Length; $i++) { $Kc[$i] = $DM::new($block, $hd); $Vc[$i] = $DM::new($block, $hd) }
    # DenseVector::new(double[]) does NOT copy: the vector uses the same array, so
    # Multiply(v, result) writes directly into our double[].
    $buf = @{}
    foreach ($spec in @(@('a', $d), @('q', $d), @('k', $d), @('v', $d), @('o', $d), @('proj', $d), @('h', $ff), @('m', $d),
                        @('logits', $vocab), @('scores', $block), @('p', $block), @('qh', $hd), @('kh', $hd), @('vh', $hd), @('oh', $hd))) {
        $arr = [double[]]::new($spec[1]); $buf[$spec[0]] = $arr; $buf[$spec[0] + 'V'] = $DV::new($arr)
    }
    return @{ K = $Kc; V = $Vc; Buf = $buf; Len = 0; Ids = [System.Collections.Generic.List[int]]::new(); LastAtt = [object[]]::new($nL * $nH); Fast = $true }
}

function Forward-TokenFast {
    # The MathNet twin of Forward-Token: the same sequence of steps (embedding ->
    # per layer LN1, attention, LN2, MLP -> final LN, Head), but the matrix multiplications
    # are done by the .NET library. Input: weights, KV-cache, one character id.
    # Output: vocab many logits (scores for the next character).
    param([hashtable]$P, [hashtable]$Cache, [int]$Id)
    $cfg = $script:Config
    $d = [int]$cfg.EmbedDim; $nL = [int]$cfg.NumLayers; $nH = [int]$cfg.NumHeads
    $hd = [int]($d / $nH); $ff = 4 * $d; $vocab = [int]$cfg.VocabSize; $block = [int]$cfg.BlockSize
    $attScale = 1.0 / [Math]::Sqrt([double]$hd); $geluC = [Math]::Sqrt(2.0 / [Math]::PI)
    $FW = $script:FastW; $B = $Cache.Buf

    if ($Cache.Len -ge $block) {
        $keepFrom = $block - [int]($block / 4)
        $keep = $Cache.Ids.GetRange($keepFrom, $Cache.Ids.Count - $keepFrom)
        $Cache.Len = 0; $Cache.Ids.Clear()
        foreach ($mtx in $Cache.K) { $mtx.Clear() }; foreach ($mtx in $Cache.V) { $mtx.Clear() }
        [Array]::Clear($B.p, 0, $B.p.Length)
        foreach ($kid in $keep) { $null = Forward-TokenFast $P $Cache $kid }
    }
    $curPos = $Cache.Len
    $Tok = $P['Tok']; $Pos = $P['Pos']
    $x = [double[]]::new($d)
    for ($j = 0; $j -lt $d; $j++) { $x[$j] = $Tok[$Id * $d + $j] + $Pos[$curPos * $d + $j] }
    $a = $B.a; $q = $B.q; $k = $B.k; $v = $B.v; $o = $B.o; $proj = $B.proj; $hh = $B.h; $m = $B.m
    $scores = $B.scores; $pArr = $B.p; $qh = $B.qh; $kh = $B.kh; $vh = $B.vh; $oh = $B.oh

    for ($l = 0; $l -lt $nL; $l++) {
        $g1 = $P["g1_$l"]; $b1 = $P["b1_$l"]; $g2 = $P["g2_$l"]; $b2 = $P["b2_$l"]
        # LN1 (PowerShell loop, 192 elements: cheap)
        $mean = 0.0; for ($j = 0; $j -lt $d; $j++) { $mean += $x[$j] }; $mean /= $d
        $var = 0.0; for ($j = 0; $j -lt $d; $j++) { $dev = $x[$j] - $mean; $var += $dev * $dev }
        $rs = 1.0 / [Math]::Sqrt($var / $d + 1e-5)
        for ($j = 0; $j -lt $d; $j++) { $a[$j] = ($x[$j] - $mean) * $rs * $g1[$j] + $b1[$j] }
        # q, k, v: three matrix-vector multiplications with MathNet
        $FW["Wq_$l"].Multiply($B.aV, $B.qV); $FW["Wk_$l"].Multiply($B.aV, $B.kV); $FW["Wv_$l"].Multiply($B.aV, $B.vV)
        # attention per head: K_h * q_h -> scores; softmax; V_h^T * p -> output
        for ($h = 0; $h -lt $nH; $h++) {
            $off = $h * $hd; $ix = $l * $nH + $h
            [Array]::Copy($k, $off, $kh, 0, $hd); $Cache.K[$ix].SetRow($curPos, $kh)
            [Array]::Copy($v, $off, $vh, 0, $hd); $Cache.V[$ix].SetRow($curPos, $vh)
            [Array]::Copy($q, $off, $qh, 0, $hd)
            $Cache.K[$ix].Multiply($B.qhV, $B.scoresV)
            $max = -1e300
            for ($u = 0; $u -le $curPos; $u++) { $s = $scores[$u] * $attScale; $scores[$u] = $s; if ($s -gt $max) { $max = $s } }
            $sum = 0.0
            for ($u = 0; $u -le $curPos; $u++) { $e = [Math]::Exp($scores[$u] - $max); $pArr[$u] = $e; $sum += $e }
            for ($u = 0; $u -le $curPos; $u++) { $pArr[$u] /= $sum }
            $att = [double[]]::new($curPos + 1); [Array]::Copy($pArr, $att, $curPos + 1); $Cache.LastAtt[$ix] = $att
            $Cache.V[$ix].TransposeThisAndMultiply($B.pV, $B.ohV)
            [Array]::Copy($oh, 0, $o, $off, $hd)
        }
        $FW["Wo_$l"].Multiply($B.oV, $B.projV)
        for ($j = 0; $j -lt $d; $j++) { $x[$j] += $proj[$j] }
        # LN2 + MLP
        $mean = 0.0; for ($j = 0; $j -lt $d; $j++) { $mean += $x[$j] }; $mean /= $d
        $var = 0.0; for ($j = 0; $j -lt $d; $j++) { $dev = $x[$j] - $mean; $var += $dev * $dev }
        $rs = 1.0 / [Math]::Sqrt($var / $d + 1e-5)
        for ($j = 0; $j -lt $d; $j++) { $a[$j] = ($x[$j] - $mean) * $rs * $g2[$j] + $b2[$j] }
        $FW["W1_$l"].Multiply($B.aV, $B.hV)
        for ($j = 0; $j -lt $ff; $j++) { $u = $hh[$j]; $hh[$j] = 0.5 * $u * (1.0 + [Math]::Tanh($geluC * ($u + 0.044715 * $u * $u * $u))) }
        $FW["W2_$l"].Multiply($B.hV, $B.mV)
        for ($j = 0; $j -lt $d; $j++) { $x[$j] += $m[$j] }
    }
    $gf = $P['gf']; $bf = $P['bf']
    $mean = 0.0; for ($j = 0; $j -lt $d; $j++) { $mean += $x[$j] }; $mean /= $d
    $var = 0.0; for ($j = 0; $j -lt $d; $j++) { $dev = $x[$j] - $mean; $var += $dev * $dev }
    $rs = 1.0 / [Math]::Sqrt($var / $d + 1e-5)
    for ($j = 0; $j -lt $d; $j++) { $a[$j] = ($x[$j] - $mean) * $rs * $gf[$j] + $bf[$j] }
    $FW['Head'].Multiply($B.aV, $B.logitsV)
    $Cache.Len = $curPos + 1; $Cache.Ids.Add($Id)
    return ,([double[]]$B.logits.Clone())
}

function New-KvCache {
    # Creating an empty KV-cache for a new conversation / generation. Per layer
    # one K and one V array (block x d), plus: Len (how many characters are in it),
    # Ids (which ones), LastAtt (the last character's attention map for the /step printout).
    param([hashtable]$P)
    if ($script:Fast) { return New-FastCache }
    $cfg = $script:Config; $d = [int]$cfg.EmbedDim; $n = [int]$cfg.BlockSize * $d
    $K = [object[]]::new($cfg.NumLayers); $V = [object[]]::new($cfg.NumLayers)
    for ($l = 0; $l -lt $cfg.NumLayers; $l++) { $K[$l] = [double[]]::new($n); $V[$l] = [double[]]::new($n) }
    return @{ K = $K; V = $V; Len = 0; Ids = [System.Collections.Generic.List[int]]::new(); LastAtt = [object[]]::new($cfg.NumLayers * $cfg.NumHeads) }
}

function Forward-Token {
    <#
      Process one character id and return next-character scores:
      one logit for each vocabulary entry.
      Update the KV-cache with data for the processed character.
      Conversion to probabilities and character selection happen later.
      This is a forward calculation only; it does not change the weights.
    #>
    param([hashtable]$P, [hashtable]$Cache, [int]$Id)
    if ($script:Fast) { return ,(Forward-TokenFast $P $Cache $Id) }
    $cfg = $script:Config
    $d = [int]$cfg.EmbedDim; $nL = [int]$cfg.NumLayers; $nH = [int]$cfg.NumHeads
    $hd = [int]($d / $nH); $ff = 4 * $d; $vocab = [int]$cfg.VocabSize; $block = [int]$cfg.BlockSize
    $attScale = 1.0 / [Math]::Sqrt([double]$hd); $geluC = [Math]::Sqrt(2.0 / [Math]::PI)

    # Window full: reprocess approximately the last quarter with positions numbered from 0.
    if ($Cache.Len -ge $block) {
        $keepFrom = $block - [int]($block / 4)   # we keep the last quarter (less frequent rebuild)
        $keep = $Cache.Ids.GetRange($keepFrom, $Cache.Ids.Count - $keepFrom)
        $Cache.Len = 0; $Cache.Ids.Clear()
        for ($l = 0; $l -lt $nL; $l++) { [Array]::Clear($Cache.K[$l], 0, $Cache.K[$l].Length); [Array]::Clear($Cache.V[$l], 0, $Cache.V[$l].Length) }
        foreach ($kid in $keep) { $null = Forward-Token $P $Cache $kid }
    }
    $curPos = $Cache.Len
    $Tok = $P['Tok']; $Pos = $P['Pos']
    $x = [double[]]::new($d)
    for ($j = 0; $j -lt $d; $j++) { $x[$j] = $Tok[$Id * $d + $j] + $Pos[$curPos * $d + $j] }

    for ($l = 0; $l -lt $nL; $l++) {
        $g1 = $P["g1_$l"]; $b1 = $P["b1_$l"]; $Wq = $P["Wq_$l"]; $Wk = $P["Wk_$l"]; $Wv = $P["Wv_$l"]; $Wo = $P["Wo_$l"]
        $g2 = $P["g2_$l"]; $b2 = $P["b2_$l"]; $W1 = $P["W1_$l"]; $W2 = $P["W2_$l"]
        $Kc = $Cache.K[$l]; $Vc = $Cache.V[$l]
        # LN1
        $mean = 0.0; for ($j = 0; $j -lt $d; $j++) { $mean += $x[$j] }; $mean /= $d
        $var = 0.0; for ($j = 0; $j -lt $d; $j++) { $dev = $x[$j] - $mean; $var += $dev * $dev }
        $rs = 1.0 / [Math]::Sqrt($var / $d + 1e-5)
        $a = [double[]]::new($d); for ($j = 0; $j -lt $d; $j++) { $a[$j] = ($x[$j] - $mean) * $rs * $g1[$j] + $b1[$j] }
        # q, k, v for the new position (vector x matrix)
        $q = [double[]]::new($d); $kb = $curPos * $d
        for ($i = 0; $i -lt $d; $i++) {
            $ai = $a[$i]; if ($ai -eq 0.0) { continue }; $wi = $i * $d
            for ($j = 0; $j -lt $d; $j++) { $q[$j] += $ai * $Wq[$wi + $j]; $Kc[$kb + $j] += $ai * $Wk[$wi + $j]; $Vc[$kb + $j] += $ai * $Wv[$wi + $j] }
        }
        # attention: the new character attends to 0..pos
        $o = [double[]]::new($d)
        for ($h = 0; $h -lt $nH; $h++) {
            $off = $h * $hd; $scores = [double[]]::new($curPos + 1); $max = -1e300
            for ($u = 0; $u -le $curPos; $u++) {
                $ki = $u * $d + $off; $s = 0.0
                for ($j = 0; $j -lt $hd; $j++) { $s += $q[$off + $j] * $Kc[$ki + $j] }
                $s *= $attScale; $scores[$u] = $s; if ($s -gt $max) { $max = $s }
            }
            $sum = 0.0; for ($u = 0; $u -le $curPos; $u++) { $e = [Math]::Exp($scores[$u] - $max); $scores[$u] = $e; $sum += $e }
            for ($u = 0; $u -le $curPos; $u++) { $scores[$u] /= $sum }
            $Cache.LastAtt[$l * $nH + $h] = $scores   # for visualization: whom the new character attended to
            for ($u = 0; $u -le $curPos; $u++) {
                $pw = $scores[$u]; $vi = $u * $d + $off
                for ($j = 0; $j -lt $hd; $j++) { $o[$off + $j] += $pw * $Vc[$vi + $j] }
            }
        }
        for ($i = 0; $i -lt $d; $i++) { $oi = $o[$i]; if ($oi -eq 0.0) { continue }; $wi = $i * $d; for ($j = 0; $j -lt $d; $j++) { $x[$j] += $oi * $Wo[$wi + $j] } }
        # LN2 + MLP
        $mean = 0.0; for ($j = 0; $j -lt $d; $j++) { $mean += $x[$j] }; $mean /= $d
        $var = 0.0; for ($j = 0; $j -lt $d; $j++) { $dev = $x[$j] - $mean; $var += $dev * $dev }
        $rs = 1.0 / [Math]::Sqrt($var / $d + 1e-5)
        $a2 = [double[]]::new($d); for ($j = 0; $j -lt $d; $j++) { $a2[$j] = ($x[$j] - $mean) * $rs * $g2[$j] + $b2[$j] }
        $hh = [double[]]::new($ff)
        for ($i = 0; $i -lt $d; $i++) { $ai = $a2[$i]; if ($ai -eq 0.0) { continue }; $wi = $i * $ff; for ($j = 0; $j -lt $ff; $j++) { $hh[$j] += $ai * $W1[$wi + $j] } }
        for ($j = 0; $j -lt $ff; $j++) { $u = $hh[$j]; $hh[$j] = 0.5 * $u * (1.0 + [Math]::Tanh($geluC * ($u + 0.044715 * $u * $u * $u))) }
        for ($i = 0; $i -lt $ff; $i++) { $hi = $hh[$i]; if ($hi -eq 0.0) { continue }; $wi = $i * $d; for ($j = 0; $j -lt $d; $j++) { $x[$j] += $hi * $W2[$wi + $j] } }
    }
    # final LN + head
    $gf = $P['gf']; $bf = $P['bf']; $Head = $P['Head']
    $mean = 0.0; for ($j = 0; $j -lt $d; $j++) { $mean += $x[$j] }; $mean /= $d
    $var = 0.0; for ($j = 0; $j -lt $d; $j++) { $dev = $x[$j] - $mean; $var += $dev * $dev }
    $rs = 1.0 / [Math]::Sqrt($var / $d + 1e-5)
    $logits = [double[]]::new($vocab)
    for ($i = 0; $i -lt $d; $i++) { $ai = (($x[$i] - $mean) * $rs) * $gf[$i] + $bf[$i]; $wi = $i * $vocab; for ($j = 0; $j -lt $vocab; $j++) { $logits[$j] += $ai * $Head[$wi + $j] } }
    $Cache.Len = $curPos + 1; $Cache.Ids.Add($Id)
    return ,$logits
}

function Sample-Logits {
    # Input: raw logits. Output: one drawn character id.
    # We divide the logits by the temperature (this "sharpens" or "flattens" the
    # distribution), turn them into probabilities with softmax, then Sample-FromProbs.
    param([double[]]$Logits)
    $Temperature = $script:Temperature; $TopK = $script:TopK
    $max = -1e300; foreach ($l in $Logits) { if ($l -gt $max) { $max = $l } }
    $probs = [double[]]::new($Logits.Length); $sum = 0.0
    for ($c = 0; $c -lt $Logits.Length; $c++) { $probs[$c] = [Math]::Exp(($Logits[$c] - $max) / $Temperature); $sum += $probs[$c] }
    for ($c = 0; $c -lt $Logits.Length; $c++) { $probs[$c] /= $sum }
    return Sample-FromProbs $probs $TopK
}

function Format-TraceChar {
    <# Makes non-printable characters visible in the trace. #>
    param([string]$ch)
    switch ($ch) { "`n" { return '\n' } "`r" { return '\r' } "`t" { return '\t' } ' ' { return [char]0xB7 } default { return $ch } }
}

function Get-NextStepTrace {
    <#
      Return a text summary and a selected character for the /step view.
      Use the already computed logits to calculate probabilities at the
      supplied temperature, then sample according to TopK.

      Show context length, the position with the largest head-averaged
      attention weight in each layer, and up to five candidate characters.
      This summarizes selected intermediate results, not the entire computation.
      An attention weight alone does not explain the output choice.

      Return @{ Lines; NextId; NextChar }.
      Do not change the cache; the caller processes the selected character.
    #>
    param([hashtable]$P, [hashtable]$Cache, [double[]]$Logits, [double]$Temp, [int]$TopK, [string]$Lang = 'hu')
    $cfg = $script:Config
    $nL = [int]$cfg.NumLayers; $nH = [int]$cfg.NumHeads; $vocab = $Logits.Length
    $curPos = $Cache.Len - 1
    $en = ($Lang -eq 'en')
    $esc = [char]27; $dim = "$esc[90m"; $cy = "$esc[36m"; $wh = "$esc[97m"; $or = "$esc[38;5;208m"; $rst = "$esc[0m"; $bold = "$esc[1m"; $grn = "$esc[32m"

    # --- softmax with temperature: logit -> probability ---
    $max = -1e300; for ($c = 0; $c -lt $vocab; $c++) { if ($Logits[$c] -gt $max) { $max = $Logits[$c] } }
    $probs = [double[]]::new($vocab); $sum = 0.0
    for ($c = 0; $c -lt $vocab; $c++) { $e = [Math]::Exp(($Logits[$c] - $max) / $Temp); $probs[$c] = $e; $sum += $e }
    for ($c = 0; $c -lt $vocab; $c++) { $probs[$c] /= $sum }
    $order = 0..($vocab - 1) | Sort-Object { $probs[$_] } -Descending | Select-Object -First 5

    $acc = [System.Collections.Generic.List[string]]::new()
    $lastCh = if ($curPos -ge 0) { Format-TraceChar $script:IdToChar[$Cache.Ids[$curPos]] } else { '?' }
    $acc.Add("$dim  " + ('-' * 52) + "$rst")
    if ($en) {
        $acc.Add("$or$bold  How the next character gets picked$rst")
        $acc.Add("$dim  The model sees $($Cache.Len) of max $($cfg.BlockSize) characters; last one: '$lastCh'$rst")
        $acc.Add("$dim  1. Each character and its place in the text become a list of $($cfg.EmbedDim) numbers.$rst")
        $acc.Add("$dim  2. $nL layers in turn rework those numbers, each looking back at the text.$rst")
        $acc.Add("$dim     This looking back is called attention.$rst")
        $acc.Add("$dim     Weight = how hard it looked there, not the chance of what comes next.$rst")
        $acc.Add("$dim     Per layer, the character it weighed most:$rst")
    } else {
        $acc.Add("$or$bold  Így dől el a következő karakter$rst")
        $acc.Add("$dim  A modell $($Cache.Len) karaktert lát (legfeljebb $($cfg.BlockSize)); az utolsó: '$lastCh'$rst")
        $acc.Add("$dim  1. Minden karakter és a helye a szövegben $($cfg.EmbedDim) számmá alakul.$rst")
        $acc.Add("$dim  2. $nL réteg gyúrja át sorban a számokat; mindegyik visszanéz a szövegre.$rst")
        $acc.Add("$dim     Ez a figyelem (attention).$rst")
        $acc.Add("$dim     A súly: mennyire figyelt oda — nem a következő karakter esélye.$rst")
        $acc.Add("$dim     Rétegenként a legerősebben figyelt karakter:$rst")
    }
    # per layer: the strongest position based on the heads' averaged attention
    $haveAtt = ($null -ne $Cache.LastAtt[0])
    if ($haveAtt) {
        for ($l = 0; $l -lt $nL; $l++) {
            $best = 0; $bestw = -1.0
            for ($u = 0; $u -le $curPos; $u++) {
                $w = 0.0
                for ($h = 0; $h -lt $nH; $h++) { $a = $Cache.LastAtt[$l * $nH + $h]; if ($a -and $u -lt $a.Length) { $w += $a[$u] } }
                $w /= $nH
                if ($w -gt $bestw) { $bestw = $w; $best = $u }
            }
            $chAt = Format-TraceChar $script:IdToChar[$Cache.Ids[$best]]
            $bar = [string]::new([char]0x2588, [int]($bestw * 20))
            $row = if ($en) { "layer {0}: mostly position {1} '{2}'; weight {3:F3}" -f $l, $best, $chAt, $bestw }
                   else     { "réteg {0}: főleg a {1} helyen álló '{2}'; súly {3:F3}" -f $l, $best, $chAt, $bestw }
            $acc.Add("$dim     $row  $cy$bar$rst")
        }
    } else {
        $acc.Add("$dim     ($(if($en){'nothing to look back at yet - the text is empty'}else{'még nincs mire visszanézni - üres a szöveg'}))$rst")
    }
    if ($en) {
        $acc.Add("$dim  3. From the result: one score for each of the $vocab known characters.$rst")
        $acc.Add("$dim  4. Scores become chances (%). Temperature $($Temp): higher = more daring.$rst")
        $acc.Add("$dim     Up to five shown, before step 5's cut; they need not add up to 100%.$rst")
    } else {
        $acc.Add("$dim  3. Az eredményből minden ismert karakter ($vocab db) kap egy pontszámot.$rst")
        $acc.Add("$dim  4. Pontszámból esély (%) lesz. Temperature $($Temp): nagyobb = merészebb.$rst")
        $acc.Add("$dim     Legfeljebb öt jelölt, még az 5. lépés előtt; összegük nem mindig 100%.$rst")
    }
    foreach ($ix in $order) {
        $c = Format-TraceChar $script:IdToChar[$ix]
        $bar = [string]::new([char]0x2588, [int]($probs[$ix] * 30))
        $acc.Add(("$dim       $wh'$c'$dim  {0:P1}  $cy$bar$rst" -f $probs[$ix]))
    }
    $nextId = Sample-FromProbs $probs $TopK
    $nextCh = Format-TraceChar $script:IdToChar[$nextId]
    $retainedMass = 1.0
    $eligibleCount = $vocab
    if ($TopK -gt 0 -and $TopK -lt $vocab) {
        $eligibleCount = $TopK
        $retainedIds = 0..($vocab - 1) | Sort-Object { $probs[$_] } -Descending | Select-Object -First $TopK
        $retainedMass = 0.0
        foreach ($candidateId in $retainedIds) { $retainedMass += $probs[$candidateId] }
    }
    $selectedChance = $probs[$nextId] / $retainedMass
    if ($en) {
        $filterLine = if ($eligibleCount -eq $vocab) { "5. Top-k is off, so every character stays in the hat for the draw." }
                      elseif ($eligibleCount -eq 1) { "5. Top-k $TopK`: only the single best character stays in the hat." }
                      else { "5. Top-k $TopK`: only the best $TopK characters stay in the hat." }
        $acc.Add("$dim  $filterLine$rst")
        $acc.Add(("$grn$bold     Drawn: '$nextCh'$rst$dim; its chance after the cut was {0:P1}.$rst" -f $selectedChance))
        $acc.Add("$dim     This character goes onto the end of the text, and the whole thing runs again.$rst")
    } else {
        $filterLine = if ($eligibleCount -eq $vocab) { "5. A top-k ki van kapcsolva: minden karakter benne marad a kalapban." }
                      elseif ($eligibleCount -eq 1) { "5. Top-k $TopK`: csak a legjobb karakter marad a kalapban." }
                      else { "5. Top-k $TopK`: csak a legjobb $TopK karakter marad a kalapban." }
        $acc.Add("$dim  $filterLine$rst")
        $acc.Add(("$grn$bold     Kihúzva: '$nextCh'$rst$dim; esélye a szűrés után {0:P1} volt.$rst" -f $selectedChance))
        $acc.Add("$dim     Ezt a szöveg végére írjuk, és az egész indul elölről.$rst")
    }
    $acc.Add("$dim  " + ('-' * 52) + "$rst")
    return @{ Lines = $acc.ToArray(); NextId = $nextId; NextChar = $script:IdToChar[$nextId] }
}

function Invoke-Generate {
    <# Feeds in the Seed text, then generates Count characters (streamed).
       This is the simplest generation loop: the Seed's characters -> cache, then
       Count times: draw a character, print it, feed it back. Characters not in
       the dictionary (which the model has never seen) are simply skipped. #>
    param([string]$Seed, [hashtable]$P, [hashtable]$Cache, [int]$Count)
    $logits = $null
    foreach ($ch in $Seed.ToCharArray()) {
        $s = [string]$ch
        if ($script:CharToId.ContainsKey($s)) { $logits = Forward-Token $P $Cache $script:CharToId[$s] }
    }
    if ($null -eq $logits) { $logits = Forward-Token $P $Cache 0 }
    for ($i = 0; $i -lt $Count; $i++) {
        $next = Sample-Logits $logits
        [Console]::Write($script:IdToChar[$next])
        $logits = Forward-Token $P $Cache $next
    }
}

function Invoke-ChatPlain {
    <#
      A line-based text continuation interface that also supports redirected I/O.
      The entered line and a newline are added to the runtime context.
      The model continues that text; training data and weights are unchanged.

      Write at most MaxTokens new characters. Two consecutive newlines can
      stop output earlier once half the length limit has been reached.
      /reset clears the context. /step 1 shows the selection of one character.
      Other commands: /temp 0.8, /tokens 300, /topk 10, /help, /q.
    #>
    param([hashtable]$P, [int]$Step)
    $cfg = $script:Config
    $en = ($script:Lang -eq 'en')
    $modelName = [IO.Path]::GetFileNameWithoutExtension($WeightsFile) -replace '^nanogpt-', '' -replace '-weights$', ''
    $isSim = $modelName -match 'orban'
    $cache = New-KvCache $P
    $temp = $Temperature; $maxTok = $MaxTokens; $topk = $TopK
    $esc = [char]27
    $dim = "$esc[90m"; $cyan = "$esc[36m"; $bold = "$esc[1m"; $white = "$esc[97m"; $yellow = "$esc[33m"; $reset = "$esc[0m"; $orange = "$esc[38;5;208m"

    if ($en) { try { [Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::InvariantCulture } catch { } }   # '.' decimal point, comma thousands separator
    try { [Console]::OutputEncoding = [Text.Encoding]::UTF8; [Console]::InputEncoding = [Text.Encoding]::UTF8 } catch { }
    try { Clear-Host } catch { }
    $title = "nanoGPT-ps  " + [char]0xB7 + "  $modelName"
    $info = if ($en) { ("{0} layers " + [char]0xB7 + " dim {1} " + [char]0xB7 + " block {2} " + [char]0xB7 + " step {3} " + [char]0xB7 + " {4:N0} params") -f $cfg.NumLayers, $cfg.EmbedDim, $cfg.BlockSize, $Step, ($P.Values | ForEach-Object { $_.Length } | Measure-Object -Sum).Sum }
             else    { ("{0} reteg " + [char]0xB7 + " dim {1} " + [char]0xB7 + " blokk {2} " + [char]0xB7 + " {3}. lepes " + [char]0xB7 + " {4:N0} parameter") -f $cfg.NumLayers, $cfg.EmbedDim, $cfg.BlockSize, $Step, ($P.Values | ForEach-Object { $_.Length } | Measure-Object -Sum).Sum }
    # Header: the pixel-art portrait (4 rows) on the left, three text rows on the right, in a rounded box.
    # The logo lines carry ANSI color codes, so their .Length is not the visible width; that width is a
    # fixed 12 columns (Get-PixelLogo emits 12-wide rows). All box padding is computed from visible widths.
    $kind = if ($isSim) { 'orban' } else { 'shakespeare' }
    $logo = Get-PixelLogo $kind
    $logoW = 12
    $simText = if ($isSim) { if ($en) { 'GENERATED TEXT - simulation, not a real quote' } else { 'GENERALT SZOVEG - szimulacio, nem valodi idezet' } } else { '' }
    $textRows = @($title, $info, $simText, '')
    $maxText = ($textRows | ForEach-Object { $_.Length } | Measure-Object -Maximum).Maximum
    $inner = 2 + $logoW + 3 + $maxText + 2   # left margin + logo + gap + widest text + right margin
    [Console]::WriteLine("$orange" + [char]0x256D + ([string]::new([char]0x2500, $inner)) + [char]0x256E + "$reset")
    for ($i = 0; $i -lt 4; $i++) {
        $t = $textRows[$i]
        $colored = switch ($i) { 0 { "$bold$white$t$reset" } 2 { "$yellow$t$reset" } default { "$dim$t$reset" } }
        $pad = ' ' * ($maxText - $t.Length + 2)
        [Console]::WriteLine("$orange" + [char]0x2502 + "$reset  " + $logo[$i] + '   ' + $colored + $pad + "$orange" + [char]0x2502 + "$reset")
    }
    [Console]::WriteLine("$orange" + [char]0x2570 + ([string]::new([char]0x2500, $inner)) + [char]0x256F + "$reset")
    if ($en) {
        [Console]::WriteLine("$dim  Enter text and the model continues it one character at a time.$reset")
        [Console]::WriteLine("$dim  /step 1 shows one selection; /help lists commands; /q exits.$reset")
    } else {
        [Console]::WriteLine("$dim  Írj be szöveget: a modell karakterenként folytatja.$reset")
        [Console]::WriteLine("$dim  /step 1: egy karakter kiválasztása; /help: parancsok; /q: kilépés.$reset")
    }

    $logits = $null   # the prediction born from the latest token; we keep it between turns too (/step)
    while ($true) {
        [Console]::Write("`n$cyan$bold" + [char]0x203A + " $reset")
        $line = [Console]::ReadLine()
        if ($null -eq $line) { break }
        $line = $line.Trim()
        if ($line -eq '') { continue }
        if ($line -match '^/(q|quit|exit)$') { break }
        if ($line -match '^/help$') {
            if ($en) {
                [Console]::WriteLine("$dim  /temp 0.8    sampling temperature; must be positive (current: $temp)")
                [Console]::WriteLine("  /tokens 300  maximum new characters (current: $maxTok)")
                [Console]::WriteLine("  /topk 10     choose among K candidates; 0 keeps all (current: $topk)")
                [Console]::WriteLine("  /step 1      show probabilities and select one new character")
                [Console]::WriteLine("  /reset       clear the current context; keep the trained weights")
                [Console]::WriteLine("  /q           exit")
                [Console]::WriteLine("  Below temperature 1, larger probabilities dominate; above 1, they become closer.")
                [Console]::WriteLine("  This interface continues text. Typing here does not train the model.$reset")
            } else {
                [Console]::WriteLine("$dim  /temp 0.8    mintavételi temperature; pozitív szám (jelenleg: $temp)")
                [Console]::WriteLine("  /tokens 300  az új karakterek felső korlátja (jelenleg: $maxTok)")
                [Console]::WriteLine("  /topk 10     választás K jelöltből; 0: minden jelölt (jelenleg: $topk)")
                [Console]::WriteLine("  /step 1      esélyek megjelenítése és egy új karakter kiválasztása")
                [Console]::WriteLine("  /reset       a kontextus törlése; a tanult súlyok megmaradnak")
                [Console]::WriteLine("  /q           kilépés")
                [Console]::WriteLine("  Temperature 1 alatt a nagyobb esélyek dominálnak, 1 felett közelebb kerülnek egymáshoz.")
                [Console]::WriteLine("  A felület szöveget folytat. Az ide írt szövegből a modell nem tanul.$reset")
            }
            continue
        }
        if ($line -match '^/temp\s+([\d.,]+)$') { $temp = [double]($Matches[1] -replace ',', '.'); [Console]::WriteLine("$dim  temperature = $temp$reset"); continue }
        if ($line -match '^/tokens\s+(\d+)$') { $maxTok = [int]$Matches[1]; [Console]::WriteLine("$dim  $(if($en){"Maximum new characters: $maxTok"}else{"Új karakterek felső korlátja: $maxTok"})$reset"); continue }
        if ($line -match '^/topk\s+(\d+)$') { $topk = [int]$Matches[1]; [Console]::WriteLine("$dim  top-k = $topk$reset"); continue }
        if ($line -match '^/reset$') { $cache = New-KvCache $P; $logits = $null; [Console]::WriteLine("$dim  $(if($en){'Context cleared. Trained weights are unchanged.'}else{'Kontextus törölve. A tanult súlyok megmaradtak.'})$reset"); continue }
        if ($line -match '^/step(\s+(\d+))?$') {
            $cnt = if ($Matches[2]) { [int]$Matches[2] } else { 1 }
            if ($null -eq $logits) { $logits = Forward-Token $P $cache $(if ($script:CharToId.ContainsKey("`n")) { $script:CharToId["`n"] } else { 0 }) }
            for ($s = 0; $s -lt $cnt; $s++) {
                $tr = Get-NextStepTrace $P $cache $logits $temp $topk $script:Lang
                foreach ($tl in $tr.Lines) { [Console]::WriteLine($tl) }
                $logits = Forward-Token $P $cache $tr.NextId
            }
            continue
        }
        if ($line -match '^/') { [Console]::WriteLine("$dim  $(if($en){'Unknown command. Use /help to list commands.'}else{'Ismeretlen parancs. A /help felsorolja a parancsokat.'})$reset"); continue }

        # --- the user's line goes into the context ---
        $logits = $null
        foreach ($ch in ($line + "`n").ToCharArray()) {
            $s = [string]$ch
            if ($script:CharToId.ContainsKey($s)) { $logits = Forward-Token $P $cache $script:CharToId[$s] }
        }
        if ($null -eq $logits) { $logits = Forward-Token $P $cache 0 }

        # --- reply streamed ---
        [Console]::Write("$orange$bold" + [char]0x25C6 + " $reset$white")
        $sw = [Diagnostics.Stopwatch]::StartNew(); $n = 0; $lastCh = ''
        for ($i = 0; $i -lt $maxTok; $i++) {
            $script:Temperature = $temp; $script:TopK = $topk
            $next = Sample-Logits $logits
            $ch = $script:IdToChar[$next]
            if ($ch -eq "`n" -and $lastCh -eq "`n" -and $n -ge [int]($maxTok / 2)) { break }   # end of paragraph = end of reply
            [Console]::Write($ch); $n++; $lastCh = $ch
            $logits = Forward-Token $P $cache $next
        }
        [Console]::Write($reset)
        if ($lastCh -ne "`n") { [Console]::WriteLine() }
        $statFmt = if ($en) { "$dim  " + [char]0xB7 + " {0} chars " + [char]0xB7 + " {1:F1} s " + [char]0xB7 + " {2:F0} char/s " + [char]0xB7 + " context {3}/{4}$reset" }
                   else      { "$dim  " + [char]0xB7 + " {0} karakter " + [char]0xB7 + " {1:F1} s " + [char]0xB7 + " {2:F0} kar/s " + [char]0xB7 + " kontextus {3}/{4}$reset" }
        [Console]::WriteLine(($statFmt -f $n, $sw.Elapsed.TotalSeconds, ($n / [Math]::Max(0.001, $sw.Elapsed.TotalSeconds)), $cache.Len, $cfg.BlockSize))
    }
    [Console]::WriteLine("$dim  $(if($en){'Exited.'}else{'Kilépés.'})$reset")
}

function Get-PixelLogo {
    <#
      Pixel-art head for the model, Claude Code style. 12 x 8 pixels, with half-block
      characters (one text row = 2 pixel rows), in 256 colors.
      Returns 4 text rows.
      (Pure decoration for the full-screen chat; nothing to do with the model.)
    #>
    param([string]$Kind)
    # palettes: letter -> 256-color
    if ($Kind -eq 'orban') {
        # short grey hair, round face, dark suit, white shirt, red tie
        $pal = @{ g = 250; s = 223; d = 52; b = 17; w = 255; r = 160 }
        $rows = @(
            '..gggggggg..',
            '.gggssssggg.',
            '.gsssssssss.',
            '.gssdssdssg.',
            '..ssssssss..',
            '..sssddsss..',
            'bbbbbwrwbbbb',
            'bbbbbbrbbbbb'
        )
    } else {
        $pal = @{ h = 94; s = 223; d = 52; w = 255; k = 236 }
        $rows = @(
            '....ssss....',
            '.hhssssssshh',
            '.hsssssssssh',
            '.hssdssdsssh',
            '.hsssssssssh',
            '.hhssdddsshh',
            'wwwwwwwwwwww',
            '.kkkkkkkkkk.'
        )
    }
    $esc = [char]27
    $out = @()
    for ($r = 0; $r -lt 8; $r += 2) {
        $line = ''
        for ($c = 0; $c -lt $rows[0].Length; $c++) {
            $t = $rows[$r][$c]; $b = $rows[$r + 1][$c]
            $tOn = $t -ne '.'; $bOn = $b -ne '.'
            if (-not $tOn -and -not $bOn) { $line += ' ' }
            elseif ($tOn -and -not $bOn) { $line += "$esc[38;5;$($pal[[string]$t])m▀$esc[0m" }
            elseif (-not $tOn -and $bOn) { $line += "$esc[38;5;$($pal[[string]$b])m▄$esc[0m" }
            else { $line += "$esc[38;5;$($pal[[string]$t])m$esc[48;5;$($pal[[string]$b])m▀$esc[0m" }
        }
        $out += $line
    }
    return ,$out
}

function Invoke-Chat {
    <#
      Full-screen version of the text continuation interface.
      Adds screen drawing, a header, and a status line to the Invoke-ChatPlain logic.
      Wrap, Render, and Read-Input handle display and input;
      skip them on a first read focused on the model.
      The -Chat entry point uses Invoke-ChatPlain for the walkthrough.
      Commands: /temp 0.8, /tokens 300, /topk 10, /step 1, /reset, /help, /q.
    #>
    param([hashtable]$P, [int]$Step)
    $cfg = $script:Config
    $en = ($script:Lang -eq 'en')
    $modelName = [IO.Path]::GetFileNameWithoutExtension($WeightsFile) -replace '^nanogpt-', '' -replace '-weights$', ''
    $kind = if ($modelName -match 'orban') { 'orban' } else { 'shakespeare' }
    $cache = New-KvCache $P
    $temp = $Temperature; $maxTok = $MaxTokens; $topk = $TopK
    $esc = [char]27
    $gray = "$esc[90m"; $bold = "$esc[1m"; $white = "$esc[97m"; $orange = "$esc[38;5;209m"; $reset = "$esc[0m"
    if ($en) { try { [Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::InvariantCulture } catch { } }
    try { [Console]::OutputEncoding = [Text.Encoding]::UTF8; [Console]::InputEncoding = [Text.Encoding]::UTF8 } catch { }
    try { [Console]::CursorVisible = $true } catch { }

    $nParams = 0; foreach ($k in $P.Keys) { $nParams += $P[$k].Length }
    $logo = Get-PixelLogo $kind
    $title = "$bold${white}nanoGPT-ps$reset $gray" + "v2.0$reset"
    $engine = if ($script:Fast) { 'MathNet' } else { if ($en) { 'pure PowerShell' } else { 'tiszta PowerShell' } }
    $line2 = if ($en) { "$gray{0} " + [char]0xB7 + " {1} layers " + [char]0xB7 + " {2} dim " + [char]0xB7 + " {3:N1}M params " + [char]0xB7 + " {4}$reset" -f $modelName, $cfg.NumLayers, $cfg.EmbedDim, ($nParams / 1e6), $engine }
             else    { "$gray{0} " + [char]0xB7 + " {1} reteg " + [char]0xB7 + " {2} dim " + [char]0xB7 + " {3:N1}M parameter " + [char]0xB7 + " {4}$reset" -f $modelName, $cfg.NumLayers, $cfg.EmbedDim, ($nParams / 1e6), $engine }
    $stepLbl = if ($en) { "step $Step" } else { "$Step. lepes" }
    $simNote = if ($kind -eq 'orban') { if ($en) { "$gray " + [char]0xB7 + " simulation, not a real quote$reset" } else { "$gray " + [char]0xB7 + " szimulacio, nem valodi idezet$reset" } } else { '' }
    $line3 = "$gray$PSScriptRoot " + [char]0xB7 + " $orange$stepLbl$reset" + $simNote
    $headerRows = 5

    # the lines of the conversation (already wrapped)
    $lines = [System.Collections.Generic.List[string]]::new()
    $GetW = { [Math]::Max(40, [Console]::WindowWidth) }
    $GetH = { [Math]::Max(15, [Console]::WindowHeight) }

    function Wrap([string]$text, [int]$width, [string]$prefix, [string]$cont) {
        $result = @()
        foreach ($para in ($text -split "`n")) {
            $cur = $prefix; $empty = $true
            foreach ($word in ($para -split ' ')) {
                if (-not $empty -and ($cur.Length + 1 + $word.Length) -gt $width) { $result += $cur; $cur = $cont + $word }
                elseif ($empty) { $cur += $word } else { $cur += ' ' + $word }
                $empty = $false
            }
            $result += $cur
        }
        return ,$result
    }

    function Render([string]$inputText, [bool]$placeholder) {
        $w = & $GetW; $h = & $GetH
        try { [Console]::Clear() } catch { }
        # header
        for ($i = 0; $i -lt 4; $i++) {
            $txt = switch ($i) { 0 { $title } 1 { $line2 } 2 { $line3 } default { '' } }
            [Console]::SetCursorPosition(0, $i); [Console]::Write('  ' + $logo[$i] + '   ' + $txt)
        }
        # conversation: the bottom part is 4 rows (right-status, line, input, line, status = 5)
        $bodyTop = $headerRows; $bodyRows = $h - $headerRows - 5
        $start = [Math]::Max(0, $lines.Count - $bodyRows)
        for ($i = 0; $i -lt $bodyRows -and ($start + $i) -lt $lines.Count; $i++) {
            [Console]::SetCursorPosition(0, $bodyTop + $i)
            $l = $lines[$start + $i]; if ($l.Length -gt $w - 1) { $l = $l.Substring(0, $w - 1) }
            [Console]::Write($l)
        }
        # bottom block
        $right = "$gray◐ temp $temp · /temp$reset"
        $rightPlain = "◐ temp $temp · /temp"
        [Console]::SetCursorPosition([Math]::Max(0, $w - $rightPlain.Length - 2), $h - 5); [Console]::Write($right)
        [Console]::SetCursorPosition(0, $h - 4); [Console]::Write($gray + ('─' * ($w - 1)) + $reset)
        [Console]::SetCursorPosition(0, $h - 3)
        if ($placeholder) { [Console]::Write("$white> $reset$gray" + $(if ($en) { 'Enter text to continue...' } else { 'Írj be szöveget a folytatáshoz...' }) + $reset) }
        else { [Console]::Write("$white> $reset$inputText") }
        [Console]::SetCursorPosition(0, $h - 2); [Console]::Write($gray + ('─' * ($w - 1)) + $reset)
        [Console]::SetCursorPosition(0, $h - 1)
        $foot = if ($en) { "  $orange" + [char]0x25B6 + [char]0x25B6 + " $modelName$reset$gray " + [char]0xB7 + " max $maxTok chars " + [char]0xB7 + " top-k $topk " + [char]0xB7 + " /help " + [char]0xB7 + " /q quit$reset" }
                else    { "  $orange" + [char]0x25B6 + [char]0x25B6 + " $modelName$reset$gray " + [char]0xB7 + " max $maxTok karakter " + [char]0xB7 + " top-k $topk " + [char]0xB7 + " /help " + [char]0xB7 + " /q kilep$reset" }
        [Console]::Write($foot)
        [Console]::SetCursorPosition(2 + $inputText.Length, $h - 3)
    }

    function Read-Input {
        $buf = ''
        Render '' $true
        while ($true) {
            $key = [Console]::ReadKey($true)
            switch ($key.Key) {
                'Enter' { return $buf }
                'Backspace' { if ($buf.Length -gt 0) { $buf = $buf.Substring(0, $buf.Length - 1); Render $buf ($buf.Length -eq 0) } }
                'Escape' { $buf = ''; Render '' $true }
                default {
                    if ($key.KeyChar -and -not [char]::IsControl($key.KeyChar)) {
                        $buf += $key.KeyChar
                        if ($buf.Length -eq 1) { Render $buf $false } else { [Console]::Write($key.KeyChar) }
                    }
                }
            }
        }
    }

    $logits = $null   # the latest prediction; we keep it between turns too (/step)
    while ($true) {
        $line = (Read-Input).Trim()
        if ($line -eq '') { continue }
        if ($line -match '^/(q|quit|exit)$') { break }
        if ($line -match '^/help$') {
            if ($en) {
                $lines.Add("$gray  /temp 0.8  sampling temperature (>0)   /tokens 300  max new characters$reset")
                $lines.Add("$gray  /topk 10   select from K candidates (0=all)   /reset  clear context$reset")
                $lines.Add("$gray  /step 1    show probabilities and select one character   /q  exit$reset")
            } else {
                $lines.Add("$gray  /temp 0.8  mintavételi temperature (>0)   /tokens 300  új karakterek korlátja$reset")
                $lines.Add("$gray  /topk 10   választás K jelöltből (0=mind)   /reset  kontextus törlése$reset")
                $lines.Add("$gray  /step 1    esélyek és egy karakter kiválasztása   /q  kilépés$reset")
            }
            continue
        }
        if ($line -match '^/temp\s+([\d.,]+)$') { $temp = [double]($Matches[1] -replace ',', '.'); continue }
        if ($line -match '^/tokens\s+(\d+)$') { $maxTok = [int]$Matches[1]; continue }
        if ($line -match '^/topk\s+(\d+)$') { $topk = [int]$Matches[1]; continue }
        if ($line -match '^/reset$') { $cache = New-KvCache $P; $logits = $null; $lines.Clear(); continue }
        if ($line -match '^/step(\s+(\d+))?$') {
            $cnt = if ($Matches[2]) { [int]$Matches[2] } else { 1 }
            if ($null -eq $logits) { $logits = Forward-Token $P $cache $(if ($script:CharToId.ContainsKey("`n")) { $script:CharToId["`n"] } else { 0 }) }
            $w = & $GetW
            for ($s = 0; $s -lt $cnt; $s++) {
                $tr = Get-NextStepTrace $P $cache $logits $temp $topk $script:Lang
                foreach ($tl in $tr.Lines) { $lines.Add($tl) }
                $logits = Forward-Token $P $cache $tr.NextId
            }
            Render '' $true
            continue
        }
        if ($line -match '^/') { $lines.Add("$gray  $(if($en){'Unknown command. Use /help to list commands.'}else{'Ismeretlen parancs. A /help felsorolja a parancsokat.'})$reset"); continue }

        $w = & $GetW
        if ($lines.Count -gt 0) { $lines.Add('') }
        foreach ($l in (Wrap $line ($w - 2) "$bold> " '  ')) { $lines.Add($l + $reset) }

        # the user's line goes into the context
        $logits = $null
        foreach ($ch in ($line + "`n").ToCharArray()) {
            $s = [string]$ch
            if ($script:CharToId.ContainsKey($s)) { $logits = Forward-Token $P $cache $script:CharToId[$s] }
        }
        if ($null -eq $logits) { $logits = Forward-Token $P $cache 0 }

        # streamed reply: we write to the end of lines, and redraw at every newline / line-length overflow
        $lines.Add("$orange◆ $reset")
        Render '' $true
        try { [Console]::CursorVisible = $false } catch { }
        $sw = [Diagnostics.Stopwatch]::StartNew(); $n = 0; $lastCh = ''; $plainLen = 2
        $h = & $GetH; $bodyRows = $h - $headerRows - 5
        for ($i = 0; $i -lt $maxTok; $i++) {
            $script:Temperature = $temp; $script:TopK = $topk
            $next = Sample-Logits $logits
            $ch = $script:IdToChar[$next]
            if ($ch -eq "`n" -and $lastCh -eq "`n" -and $n -ge [int]($maxTok / 2)) { break }
            $n++; $lastCh = $ch
            if ($ch -eq "`n" -or $plainLen -ge ($w - 2)) {
                $lines.Add('  '); $plainLen = 2
                if ($ch -ne "`n") { $lines[$lines.Count - 1] += $ch; $plainLen++ }
                Render '' $true
            } else {
                $lines[$lines.Count - 1] += $ch; $plainLen++
                $row = $headerRows + [Math]::Min($lines.Count, $bodyRows) - 1
                [Console]::SetCursorPosition($plainLen - 1, $row); [Console]::Write($ch)
            }
            $logits = Forward-Token $P $cache $next
        }
        $stFmt = if ($en) { "$gray  " + [char]0xB7 + " {0} chars " + [char]0xB7 + " {1:F1} s " + [char]0xB7 + " {2:F1} char/s " + [char]0xB7 + " context {3}/{4}$reset" }
                 else      { "$gray  " + [char]0xB7 + " {0} karakter " + [char]0xB7 + " {1:F1} s " + [char]0xB7 + " {2:F1} kar/s " + [char]0xB7 + " kontextus {3}/{4}$reset" }
        $lines.Add(($stFmt -f $n, $sw.Elapsed.TotalSeconds, ($n / [Math]::Max(0.001, $sw.Elapsed.TotalSeconds)), $cache.Len, $cfg.BlockSize))
        try { [Console]::CursorVisible = $true } catch { }
    }
    try { [Console]::Clear() } catch { }
    Write-Host $(if ($en) { 'Exited.' } else { 'Kilépés.' })
}

# ============================================================================
#  PART 7: MAIN PROGRAM
# ============================================================================
<#
  Select a mode in this order:
    -AsLibrary: return after loading definitions and initial state.
    -GradCheck: run a gradient check on a small model.
    -Train: start or resume training.
    Otherwise: load weights and generate text.
  -Chat opens an interactive interface for generation.
  -Prompt supplies the starting text for direct generation.
#>

if (-not $AsLibrary) {
try { [Console]::ForegroundColor = [ConsoleColor]::Green } catch { }
if ($script:Lang -eq 'en') {
[Console]::WriteLine(@"
  ╔══════════════════════════════════════════════════════════╗
  ║   nanoGPT-ps v2  ·  full training in pure PowerShell      ║
  ╚══════════════════════════════════════════════════════════╝
"@)
} else {
[Console]::WriteLine(@"
  ╔══════════════════════════════════════════════════════════╗
  ║   nanoGPT-ps v2  ·  teljes tanitas tiszta PowerShellben   ║
  ╚══════════════════════════════════════════════════════════╝
"@)
}
try { [Console]::ResetColor() } catch { }
}

if ($AsLibrary) { return }   # when dot-sourced we only load the functions (used by viz.ps1)
if ($GradCheck) { Invoke-GradCheck; return }
if ($Train) { Invoke-Train $CorpusFile $TrainSteps; return }

$en = ($script:Lang -eq 'en')
if (-not (Test-Path $WeightsFile)) {
    if ($en) {
        Write-Host "Weight file not found: $WeightsFile" -ForegroundColor Yellow
        Write-Host 'To use an existing model, pass -WeightsFile with the path to its JSON file.'
        Write-Host 'To train a new model, pass -Train -CorpusFile <text-file> -WeightsFile <new-json-file>.'
    } else {
        Write-Host "Nem található a súlyfájl: $WeightsFile" -ForegroundColor Yellow
        Write-Host 'Meglévő modellhez add meg a JSON-fájl útját a -WeightsFile kapcsolóval.'
        Write-Host 'Új modell tanításához: -Train -CorpusFile <szövegfájl> -WeightsFile <új-json-fájl>.'
    }
    return
}
Write-Host $(if ($en) { "Loading weights: $WeightsFile ..." } else { "Sulyok betoltese: $WeightsFile ..." }) -ForegroundColor Cyan
$ck = Load-Checkpoint $WeightsFile
Initialize-Fast $ck.Params
if ($en) {
    Write-Host ("OK. {0} layers, dim {1}, block {2}, step {3}, {4}" -f $script:Config.NumLayers, $script:Config.EmbedDim, $script:Config.BlockSize, $ck.Step, $(if ($script:Fast) { 'fast path (MathNet)' } else { 'pure PowerShell' })) -ForegroundColor Green
} else {
    Write-Host ("OK. {0} reteg, dim {1}, blokk {2}, {3}. lepes, {4}" -f $script:Config.NumLayers, $script:Config.EmbedDim, $script:Config.BlockSize, $ck.Step, $(if ($script:Fast) { 'gyors ut (MathNet)' } else { 'tiszta PowerShell' })) -ForegroundColor Green
}
if ($Chat) {
    # The scrollable interface keeps longer /step output available for review.
    Invoke-ChatPlain $ck.Params $ck.Step
    return
}
Write-Host $(if ($script:Lang -eq 'en') { "`n--- Text continuation ---" } else { "`n--- Szövegfolytatás ---" }) -ForegroundColor Cyan
if ($Prompt -eq '') { $Prompt = "`n" }
try { [Console]::ForegroundColor = [ConsoleColor]::Yellow } catch { }
[Console]::Write($Prompt)
try { [Console]::ForegroundColor = [ConsoleColor]::White } catch { }
$count = if ($Endless) { [int]::MaxValue } else { $MaxTokens }
Invoke-Generate $Prompt $ck.Params (New-KvCache $ck.Params) $count
try { [Console]::ResetColor() } catch { }
[Console]::WriteLine()
