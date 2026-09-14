<#
.SYNOPSIS
    Karakterenként szöveget folytató GPT, PowerShellben megírva.
.DESCRIPTION
    A modell kap egy szövegrészletet, és megbecsüli a következő karakter
    valószínűségét. Kiválasztunk egy karaktert, hozzáírjuk a szöveghez,
    majd megismételjük a lépést. Itt egy token egy karakter.

    Tanításkor a szöveg valódi folytatása is rendelkezésre áll. Ebből
    számoljuk a loss-t, majd módosítjuk a modell tanulható számait,
    a súlyokat. Generáláskor a súlyok változatlanok.

    A forrás olvasásához érdemes ismerni a PowerShell tömbjeit,
    ciklusait és függvényeit. A GPT fogalmait a számítások mellett
    vezetjük be. PowerShell 7 szükséges a futtatáshoz.

    Első alkalommal próbáld ki a generálást egy kész súlyfájllal.
    A tanítás PowerShellben lassú; megértéséhez először egy kis modell
    néhány lépését érdemes követni.

    Külső könyvtár nélkül is fut. Generáláskor a mellékelt MathNet
    könyvtár gyorsíthatja a számítást; a -NoFast ezt kikapcsolja.
.PARAMETER Train
    Tanítás indítása. Létező súlyfájlból folytatja a tanítást.
    A mentett Adam-állapotot is betölti, ha rendelkezésre áll.
.PARAMETER TrainSteps
    Az ebben a futásban végrehajtott tanítási lépések száma. Alapérték: 3000.
.PARAMETER BatchSize
    Egy súlyfrissítéshez felhasznált szövegablakok száma. Alapérték: 8.
.PARAMETER Threads
    A párhuzamos feladatok felső korlátja. Alapérték: a .NET által
    jelentett logikai processzorszám. A BatchSize ettől független.
.PARAMETER GradCheck
    A kézzel számolt gradienst numerikus közelítéssel ellenőrzi
    egy kis modellen, súlycsoportonként négy kiválasztott elemen.
.PARAMETER Layers
    Transformer-rétegek száma új modellnél. Alapérték: 3.
.PARAMETER Dim
    Egy pozíció vektorának hossza új modellnél. Alapérték: 64.
    Oszthatónak kell lennie a Heads értékével.
.PARAMETER Heads
    Attention-fejek száma rétegenként új modellnél. Alapérték: 4.
.PARAMETER Block
    Egy szövegablak legnagyobb hossza új modellnél. Alapérték: 64.
    Folytatáskor a modell méretei a súlyfájlból származnak.
.PARAMETER Lang
    A felület nyelve: hu vagy en. A tanult szöveg nyelvét nem változtatja meg.
.PARAMETER MaxTokens
    Legfeljebb ennyi új karaktert generál. Alapérték: 200.
.PARAMETER Temperature
    A mintavételi eloszlás alakját szabályozó pozitív szám.
    1 alatt a nagyobb esélyek dominálnak, 1 felett az esélyek közelebb kerülnek.
.PARAMETER TopK
    Csak a K legvalószínűbb karakterből választ. A 0 minden karaktert megtart.
.PARAMETER WeightsFile
    A betöltendő vagy tanításkor mentendő súlyfájl elérési útja.
.PARAMETER CorpusFile
    A tanításhoz használt szövegfájl elérési útja.
.EXAMPLE
    pwsh -File .\nanogpt-ps.ps1 -Lang hu -Prompt "ROMEO:" -MaxTokens 80
.EXAMPLE
    pwsh -File .\nanogpt-ps.ps1 -Lang hu -Chat
    A felületen a /step 1 egy karakter kiválasztását mutatja meg.
.EXAMPLE
    pwsh -File .\nanogpt-ps.ps1 -Lang hu -GradCheck
.EXAMPLE
    pwsh -File .\nanogpt-ps.ps1 -Lang hu -Train -Layers 1 -Dim 16 -Heads 2 -Block 16 -BatchSize 1 -Threads 1 -TrainSteps 20 -WeightsFile .\demo-weights.json
    Új demo-weights.json fájllal kis modellt hoz létre. A meglévő fájlból folytatja.
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
$script:Lang = $Lang   # 'hu' vagy 'en' - a chat feluletek nyelve
# Angol módban mindenhol pont a tizedesjel (a gradcheck, a tanítás és a generálás számainál is),
# nem csak a chat felületén, hogy a dokumentált -Lang en parancsok minden területi beállításnál ugyanúgy jelenjenek meg.
if ($script:Lang -eq 'en') { try { [Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::InvariantCulture } catch { } }

<#
  Olvasási útvonal

  1. Szövegfolytatás: Invoke-Generate.
     Figyeld meg, hogyan lesz a kiválasztott karakterből a következő bemenet.
  2. Egy karakter feldolgozása: Forward-Token.
     Embedding -> transformer-rétegek -> végső LayerNorm -> Head -> logits.
  3. Karakterválasztás: Sample-Logits és Sample-FromProbs.
     Logits -> temperature + softmax -> top-k -> mintavétel.
  4. Tanítás: Compute-SeqGrad, először csak a FORWARD rész.
     Ugyanaz a modell, de most a valódi következő karaktert is ismerjük.
  5. Súlyfrissítés: BACKWARD, majd Invoke-Train és az Adam blokk.

  A függvénynevekre keresve ebben a sorrendben is olvashatod a fájlt.
  A mentés, a párhuzamosítás és a képernyőrajzolás későbbre hagyható.

  Rövid jelölésmagyarázat:
    token / id: itt egy karakter, illetve annak egész számú azonosítója;
    vocab: az ismert karakterek száma, a szóközt és újsort is beleértve;
    d / Dim: az egy pozícióhoz tartozó vektor elemeinek száma;
    T / seqLen: az egyszerre feldolgozott bemeneti pozíciók száma;
    nH / Heads: attention-fejek száma; hd = d / nH;
    nL / Layers: transformer-rétegek száma;
    P: tanulható paramétertáblák; G: a hozzájuk tartozó gradiensek.

  Egy vektor számsor, egy mátrix sorokba és oszlopokba rendezett számtábla.
  A kód a mátrixokat egydimenziós tömbként tárolja:
  M[i,j] helye A[i*oszlopszám+j]. Az indexek 0-tól indulnak.
#>

# ============================================================================
#  1. RESZ: A MODELL SULYAI
#     Minden matrix egy sima double[] (sor-folytonos: M[i,j] = A[i*cols+j]).
#     A súlyok név szerint érhetők el: $P['Wq_2'] a 2-es indexű réteg query-mátrixa.
# ============================================================================
<#
  A súlyok a modell tanulható számai. Tanításkor ezeket módosítjuk,
  hogy a modell nagyobb valószínűséget adjon a szöveg valódi folytatásának.
  Generáláskor a súlyok változatlanok.

  Többféle táblába rendezzük őket. A Tok minden karakterhez egy
  tanulható számsort tárol. A rétegek ezekkel a számsorokkal dolgoznak.

  A Head az utolsó lépéshez tartozó súlymátrix neve. A rétegek után
  kapott számsorból minden ismert karakterhez egy pontszámot számol.
  Az "alm" után például ilyen pontszámok készülhetnének:
    a: 3.2, e: 1.1, z: -0.8  (szemléltető példa).
  A nagyobb pontszámú karakterből a softmax lépésben nagyobb esély lesz.
  A karaktert ezekből az esélyekből választjuk ki.
  Ez a Head nem ugyanaz, mint a rétegeken belüli attention-fej.

  A következő két függvény a táblák méretét és kezdőértékeit adja meg.
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
  A súlytáblák méreteit adja vissza: név -> (sorok, oszlopok).
  Például 65 karakter és Dim=64 esetén a Tok táblája 65 x 64 szám.
  Egy sor egy karakterhez tartozik.
  Ezt a méretlistát használjuk a súlyok létrehozásához és ellenőrzéséhez.
#>
    param([hashtable]$Cfg)
    $d = $Cfg.EmbedDim; $ff = 4 * $d
    $shapes = [ordered]@{
        Tok = @($Cfg.VocabSize, $d)   # token embedding
        Pos = @($Cfg.BlockSize, $d)   # pozicio embedding
    }
    for ($l = 0; $l -lt $Cfg.NumLayers; $l++) {
        $shapes["g1_$l"] = @(1, $d);   $shapes["b1_$l"] = @(1, $d)   # layernorm 1
        $shapes["Wq_$l"] = @($d, $d);  $shapes["Wk_$l"] = @($d, $d)
        $shapes["Wv_$l"] = @($d, $d);  $shapes["Wo_$l"] = @($d, $d)
        $shapes["g2_$l"] = @(1, $d);   $shapes["b2_$l"] = @(1, $d)   # layernorm 2
        $shapes["W1_$l"] = @($d, $ff); $shapes["W2_$l"] = @($ff, $d) # MLP
    }
    $shapes['gf'] = @(1, $d); $shapes['bf'] = @(1, $d)              # vegso layernorm
    $shapes['Head'] = @($d, $Cfg.VocabSize)
    return $shapes
}

function New-GptParams {
    <#
  Létrehozza a súlytáblákat egy még be nem tanított modellhez.
  A mátrixok elemei 0 átlagú, 0.02 szórású normális eloszlásból indulnak.
  Az eltérő kezdőértékek lehetővé teszik, hogy az egységek eltérő
  mintázatokat tanuljanak.

  Kivételek: a LayerNorm skálái 1-ről, eltolásai 0-ról indulnak.
  A residual ágba író Wo és W2 mátrixok szórását ezen felül
  1/sqrt(2*NumLayers) szorzóval csökkentjük.

  A Seed a véletlenszám-generátor kezdőértéke. Azonos konfigurációval,
  ugyanebben a környezetben megismételhetővé teszi az inicializálást.
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
                # Box-Muller: ket egyenletes -> egy normal eloszlasu szam
                $u1 = 1.0 - $rng.NextDouble(); $u2 = $rng.NextDouble()
                $a[$i] = $std * [Math]::Sqrt(-2.0 * [Math]::Log($u1)) * [Math]::Cos(2.0 * [Math]::PI * $u2)
            }
        }
        $P[$name] = $a
    }
    return $P
}

# ============================================================================
#  2. RESZ: FORWARD + BACKWARD EGY SZOVEGABLAKRA
#
#     Ez a script szive. Egyetlen, onallo fuggveny, mert a parhuzamos
#     szalakba (runspace) csak ezt adjuk at szovegkent. Ezert NEM hiv
#     mas fuggvenyt; a matrixszorzasok helyi scriptblockok.
#
#     Bemenet:  $Ids = T+1 karakter-id. Input = Ids[0..T-1], cel = Ids[1..T]
#     Kimenet:  @{ Loss = atlagos cross-entropy; Grads = nev -> double[] }
#
#     Képletek: a lineáris vetítésekben nincs bias; a LayerNormnak van tanulható eltolása.
#       x0 = Tok[id] + Pos[t]
#       minden retegben:
#         a  = LN1(x)                 q = a Wq, k = a Wk, v = a Wv
#         att = softmax(q k^T / sqrt(hd), kauzalis)     o = att v
#         x  = x + o Wo
#         a2 = LN2(x)                 h = a2 W1,  gh = GELU(h)
#         x  = x + gh W2
#       af = LNf(x);  logits = af Head;  loss = -log softmax(logits)[cel]
#
#     A backward ugyanez visszafele, a lancszabaly szerint.
# ============================================================================
<#
  Egy szövegablakból tanítópárokat készítünk. Például:

    szöveg:    a l m a
    bemenet:   a l m
    cél:       l m a

  A három pozíció feladata: a -> l, al -> m, alm -> a.
  A kauzális maszk gondoskodik róla, hogy egy pozíció csak saját
  magát és a korábbi pozíciókat használhassa. Így tanításkor sem
  olvashatja ki előre a helyes választ a későbbi bemenetből.

  Forward: minden pozícióhoz következőkarakter-valószínűségeket
  számolunk. A loss azt méri, mennyire kevés esélyt kapott a valódi
  folytatás. Ezt a számot átlagoljuk a pozíciókra.

  Backward: kiszámoljuk, hogyan változna a loss az egyes súlyok kis
  változtatására. Ez a gradiens. Egy kis delta változtatás hatása
  közelítőleg gradiens * delta. A súlyokat itt még nem módosítjuk;
  ezt az Invoke-Train végzi az Adam optimalizálóval.

  Első olvasáskor a FORWARD részt kövesd a loss kiszámításáig.
  A BACKWARD részhez a generálás megértése után is visszatérhetsz.
  A mátrixműveletek helyben vannak definiálva, mert a párhuzamos
  feladatok ezt a függvényt kapják meg önállóan, szövegként.
#>

function Compute-SeqGrad {
    param([hashtable]$P, [int[]]$Ids, [hashtable]$Cfg, [bool]$NeedGrad = $true)

    $d = [int]$Cfg.EmbedDim; $nL = [int]$Cfg.NumLayers; $nH = [int]$Cfg.NumHeads
    $hd = [int]($d / $nH); $ff = 4 * $d; $vocab = [int]$Cfg.VocabSize
    $seqLen = $Ids.Length - 1
    $attScale = 1.0 / [Math]::Sqrt([double]$hd)
    $geluC = [Math]::Sqrt(2.0 / [Math]::PI)

    # A mátrixszorzás egy bemeneti számsorból új számsort készít.
    # Példa: [2, 3] és egy két sorból álló mátrix:
    #   [1, 4]
    #   [5, 6]
    # Eredmény: [2*1 + 3*5, 2*4 + 3*6] = [17, 26].
    # Minden kimeneti elem a bemenet súlyozott összege.
    # MM: szorzás; MMT: szorzás a második mátrix transzponáltjával;
    # ATB: az első mátrix transzponáltjával számolt hozzájárulás hozzáadása G-hez.
    # Transzponáláskor felcseréljük a sor- és oszlopindexet: B^T[i,j] = B[j,i].
    # A nulla bemeneti elemek kihagyása csak a felesleges szorzásokat spórolja meg.
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
    # G(ca x cb) += A(ra x ca)^T * B(ra x cb)   (sulygradiens)
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
    # Az Ids[t] karakterazonosítóval kiválasztunk egy sort a Tok táblából.
    # Ez d darab tanulható szám; az azonosító csak a sor sorszáma.
    # Hozzáadjuk a t. pozícióhoz tartozó sort a Pos táblából.
    # Példa két dimenzióval: [0.2, -0.1] + [0.0, 0.3] = [0.2, 0.2].
    # A valódi modellben minden pozícióhoz d szám tartozik, az x tömbben.
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
        # Minden pozíció d számából kivonjuk az átlagukat, majd elosztjuk
        # őket sqrt(variancia + 1e-5) értékével. A kis konstans megakadályozza
        # a nullával osztást. Ettől a különböző bemenetek skálája egyenletesebb.
        # A normalizált értékeket a tanulható g skálázza és b eltolja.
        # Nincs rögzített alsó vagy felső értékhatár.
        # Az xhat és rstd köztes eredmények a backward számításhoz kellenek.
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

        # --- Kauzalis multi-head attention ---
        # att[h, t, u] = mennyire figyel a t. karakter az u. karakterre (u <= t)
        #
        # Az attention más pozíciók információját keveri az aktuális pozícióhoz.
        # Ugyanabból a normalizált bemenetből három tanult vetület készül:
        #   q: ezzel hasonlítjuk az aktuális pozíciót a többihez;
        #   k: ehhez hasonlítjuk a q-t az egyes elérhető pozíciókon;
        #   v: ezeket a vektorokat összegezzük a kapott súlyokkal.
        # A q és k skalárszorzata adja a pontszámot, sqrt(hd)-vel skálázva.
        # A softmax a pontszámokból nemnegatív, összesen 1-et adó súlyokat készít.
        # Példa: [0.2, 0.3, 0.5] esetén az eredmény 0.2*v0 + 0.3*v1 + 0.5*v2.
        # Ezek attention-súlyok; a következő karakter esélyeit később számoljuk.
        # A t. pozíció csak a 0..t pozíciókat használja, önmagát is beleértve.
        # A későbbieket kizárjuk, mert generáláskor még nem állnának rendelkezésre.
        # Minden fej a q, k és v egy hd hosszú szeletén külön súlyozást számol.
        # A fejek eltérő mintázatokat tanulhatnak; a szerepük nincs előre kiosztva.
        # A softmax előtt kivonjuk a legnagyobb pontszámot. Ez nem változtatja
        # meg az eloszlást, de elkerüli a túl nagy számok exponenciálását.
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
        # Az MLP minden pozíciót külön dolgoz fel, azonos súlymátrixokkal.
        # W1 a d hosszú vektort 4*d hosszúra alakítja, a GELU elemenként
        # nemlineáris függvényt alkalmaz, W2 pedig d hosszúra alakítja vissza.
        # GELU nélkül ez a két mátrixszorzás egyetlen szorzássá összevonható lenne.
        # A teljes transformerben más nemlineáris műveletek is vannak.
        # A GELU-t itt a lent látható tanh-közelítéssel számoljuk.
        # A residual összeadás: x2 = x1 + m. A bemenet közvetlenül is
        # eljut a következő réteghez, a számolt m hozzájárulással együtt.
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

    # --- Vegso LayerNorm + Head + softmax + loss (MINDEN poziciora) ---
    # A végső LayerNorm után a Head minden lehetséges karakterhez
    # egy pontszámot ad. Ezek a logits: még nem valószínűségek.
    # A softmaxból kapott valószínűségek összege pozíciónként 1.
    # A loss ezen a pozíción: -ln(p), ahol p a valódi következő karakter esélye.
    # Példák: p=0.5 -> loss=0.693; p=0.1 -> 2.303; p=0.01 -> 4.605.
    # Minél több esélyt kap a helyes karakter, annál kisebb ez a veszteség.
    # A függvény a pozíciók loss-át átlagolja. 65 karakterre egyenletes
    # eloszlást adva a loss ln(65), azaz körülbelül 4.174.
    # Ez az adott ablak eredménye. Az új szövegekre való általánosítást
    # külön, tanításhoz nem használt szövegen kell mérni.
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
    # A dlogits, daf és dx a loss adott köztes eredmény szerinti deriváltja.
    # Pozitív érték esetén az adott elem kis növelése helyben növelné a loss-t;
    # negatív értéknél csökkentené. A jelölésben d deriváltat jelent,
    # de az önálló $d változó továbbra is a modell dimenziója.
    # A $G az egyes súlytáblák gradiensét gyűjti, a $P-vel azonos kulcsokkal.
    $G = @{}

    # A softmax + cross-entropy logits szerinti deriváltja: (p - cél) / T.
    # A célvektorban a helyes karakternél 1 áll, minden más helyen 0.
    # Példa: p=[0.2, 0.5, 0.3], helyes a második karakter.
    # Ekkor p-cél=[0.2, -0.5, 0.3]. Ezt még T-vel osztjuk,
    # mert a loss T pozíció átlagaként lett kiszámolva.
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

    # Egy bemeneti elem változása az átlagot és a varianciát is módosítja,
    # így ugyanazon pozíció többi normalizált elemére is hat.
    # A backward képlet két átlagtagja ezt a közös függést veszi figyelembe.
    # Ugyanezt a képletet használjuk később az LN2 és LN1 visszaszámításához.
    # LayerNorm backward (kozos keplet):
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
        # GELU derivalt
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
        # Az o eredmény a v vektorok súlyozott összege. Visszafelé ezért
        # mind a v értékeihez, mind az attention-súlyokhoz számolunk gradienst.
        # A v felé haladó hozzájárulást az adott attention-súly szorozza.
        # Az attention-súlyok gradiense a softmaxon keresztül jut a q és k felé.
        # Végül a Wq, Wk és Wv súlymátrixok gradienseit is összegyűjtjük.
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

    # --- Embedding gradiens ---
    # Ugyanaz a karakter több pozícióban is előfordulhat. Mindegyik
    # előfordulás a Tok ugyanazon sorát használta, ezért a hozzájuk tartozó
    # gradienseket összeadjuk: ezért += áll itt értékadás helyett.
    # A hozzájárulások erősíthetik vagy részben kiolthatják egymást.
    # A pozícióvektorok gradienseit a Pos megfelelő soraiba gyűjtjük.
    $dTok = [double[]]::new($vocab * $d); $dPos = [double[]]::new([int]$Cfg.BlockSize * $d)
    for ($t = 0; $t -lt $seqLen; $t++) {
        $ti = $Ids[$t] * $d; $xi = $t * $d
        for ($j = 0; $j -lt $d; $j++) { $dTok[$ti + $j] += $dx[$xi + $j]; $dPos[$xi + $j] += $dx[$xi + $j] }
    }
    $G['Tok'] = $dTok; $G['Pos'] = $dPos

    return @{ Loss = $loss; Grads = $G }
}

# ============================================================================
#  3. RESZ: MENTES / BETOLTES
# ============================================================================
<#
  A checkpoint a tanítás közben mentett állapot.
  A JSON tartalmazza a modell méreteit, a karakterazonosítókat,
  a súlyokat, az Adam M és V tömbjeit, valamint a lépésszámot.
  Ebből folytatható a tanítás, és ebből töltjük be a modellt generáláshoz.

  A véletlenszám-generátor állapotát nem mentjük, ezért a folytatás
  nem feltétlenül ugyanazokat a tanítóablakokat választja, mint
  egy megszakítás nélküli futás. A tanulási ráta további ütemezését
  a folytatáskor kért lépésszám is befolyásolja.
#>

function Save-Checkpoint {
    # A súlyokat, az Adam állapotát és a lépésszámot JSON-fájlba menti.
    # Először a .tmp fájlt írjuk meg, majd azzal cseréljük le a célfájlt.
    # Így a JSON írása közben még a korábbi checkpoint marad a célhelyen.
    param([string]$Path, [hashtable]$P, [hashtable]$Adam, [int]$Step)
    $out = [ordered]@{
        Version  = 2
        Config   = $script:Config
        Step     = $Step
        # csak a Vocab lista (id -> karakter); a CharToId-t nem mentjuk, mert a JSON-olvaso
        # nem tur csak kis/nagybetuben elterо kulcsokat
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
    # Bemenet: a JSON fájl útvonala. Kimenet: @{ Params; Adam; Step } - a Save-Checkpoint
    # tükörképe. Mellékhatásként beállítja a globális Config-ot és a karakter-szótárat
    # (CharToId / IdToChar), mert ezek nélkül a betöltött súlyok használhatatlanok.
    # Elutasítja a régi (csak-Head) formatumú fájlt, mert abban csak a Head réteg volt.
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
    # FONTOS: a PowerShell hashtable kulcsa kis/nagybetu-erzeketlen ('a' == 'A'),
    # ezert a karakter->id tabla case-erzekeny Dictionary.
    $script:CharToId = [System.Collections.Generic.Dictionary[string,int]]::new([StringComparer]::Ordinal); $script:IdToChar = @{}
    if ($json.PSObject.Properties.Name -contains 'Vocab') {
        $vocabList = @($json.Vocab)
        for ($i = 0; $i -lt $vocabList.Count; $i++) { $script:CharToId[[string]$vocabList[$i]] = $i; $script:IdToChar[$i] = [string]$vocabList[$i] }
    } else {
        # Regi checkpoint (a case-osszeolvadas hibaval tanitva): a modell kisbetus szoveget tanult,
        # ezert a kiiras is kisbetus, a prompt betuit pedig kisbetusitjuk.
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
#  4. RESZ: TANITAS (Adam, parhuzamos batch)
# ============================================================================
<#
  Egy tanítási lépés:
    1. Választunk BatchSize darab részletet a tanítószövegből.
       Mindegyik BlockSize+1 karakter: BlockSize bemenet és az eltolt célok.
    2. Mindegyikre kiszámoljuk a loss-t és a gradienseket.
       Az ablakok ugyanazokat a súlyokat olvassák, egymástól függetlenül.
    3. Átlagoljuk a gradienseket. Ha az összesített gradiens normája
       nagyobb 1-nél, arányosan visszaskálázzuk.
    4. Az Adam a gradiensek mozgóátlagai alapján módosítja a súlyokat.
    5. Kiírjuk a mért értékeket, és időnként checkpointot mentünk.

  A batch egy súlyfrissítéshez használt ablakcsoport.
  A Threads a párhuzamosság felső korlátja, nem a batch mérete.
  A kiírt loss a frissítés előtt feldolgozott tanítóablakokon mért érték.
#>

function Invoke-Train {
    # Bemenet: a korpusz fájl és a futtatandó lépések száma. Kimenet: nincs
    # visszatérési érték; a súlyfájl (checkpoint) frissül a lemezen, a
    # konzolra pedig a haladás megy. Ha már van súlyfájl, onnan folytatja.
    param([string]$Corpus, [int]$Steps)
    Write-Host $(if ($script:Lang -eq 'en') { "`n=== Training: updating weights in every layer ===" } else { "`n=== Tanítás: minden réteg súlyai frissülnek ===" }) -ForegroundColor Yellow
    $text = [System.IO.File]::ReadAllText($Corpus)

    $step0 = 0
    if (Test-Path $WeightsFile) {
        Write-Host $(if ($script:Lang -eq 'en') { "Resuming training from: $WeightsFile" } else { "Tanítás folytatása ebből a mentésből: $WeightsFile" }) -ForegroundColor Yellow
        $ck = Load-Checkpoint $WeightsFile
        $P = $ck.Params; $adam = $ck.Adam; $step0 = $ck.Step
        # GPU-s (train_gpu.py) export: nincs Adam-allapot, nullarol indul
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

    # A per-ablak forward+backward fuggveny szovegkent, hogy a szalak megkapjak
    $fnCode = ${function:Compute-SeqGrad}.ToString()
    $rng = [Random]::new()
    $block = $cfg.BlockSize
    $beta1 = 0.9; $beta2 = 0.99; $eps = 1e-8
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lossWindow = [System.Collections.Generic.Queue[double]]::new()
    $paramNames = @($P.Keys)

    for ($step = $step0 + 1; $step -le $step0 + $Steps; $step++) {
        # --- batch: BatchSize veletlen ablak ---
        # A szövegből véletlen helyen kivágunk block+1 karaktert: az első block a
        # bemenet, az eggyel eltolt block a "helyes válasz" (mindig a következő betű).
        $batch = [object[]]::new($BatchSize)
        for ($b = 0; $b -lt $BatchSize; $b++) {
            $start = $rng.Next(0, $text.Length - $block - 1)
            $ids = [int[]]::new($block + 1)
            for ($i = 0; $i -le $block; $i++) { $ids[$i] = $script:CharToId[[string]$text[$start + $i]] }
            $batch[$b] = $ids
        }

        # --- forward+backward parhuzamosan (a sulyok referenciakent mennek at) ---
        # Minden ablak külön szálon fut, mert egymástól függetlenek: mindegyik csak
        # OLVASSA a súlyokat és a saját gradiensét adja vissza. A szálak nem írnak
        # közös adatot, ezért nem kell zárolás. A $using: a PowerShell módja, hogy
        # a szálon belül a külső változókat elérjük.
        if ($Threads -gt 1) {
            $results = $batch | ForEach-Object -ThrottleLimit $Threads -Parallel {
                $f = [scriptblock]::Create($using:fnCode)
                & $f $using:P $_ $using:cfg $true
            }
        } else {
            $results = foreach ($ids in $batch) { Compute-SeqGrad $P $ids $cfg $true }
        }

        # --- gradiensek atlagolasa + globalis norma (clip 1.0) ---
        # A 8 ablak gradiensét összeadjuk és elosztjuk 8-cal. A "clip" biztonsági
        # fék: ha a gradiens összhossza (norma) 1-nél nagyobb, arányosan lekicsinyítjük.
        # Így egy-egy furcsa szövegrészlet nem tud óriási, elrontó lépést okozni.
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
        # Egyszerű gradienslépésnél a súlyból lr * gradiens értéket vonnánk ki.
        # Az Adam súlyonként két exponenciális mozgóátlagot tart:
        #   m: a gradiens átlaga, amely a korábbi irányokat is figyelembe veszi;
        #   v: a gradiens négyzetének átlaga, amely a lépés skálázásához kell.
        # A v nem a modell bizonytalanságát méri.
        # A bc1 és bc2 a nulláról induló mozgóátlagok torzítását korrigálja.
        # A frissítés a korrigált m / (sqrt(korrigált v) + eps) értéket használja,
        # megszorozva az aktuális lr tanulási rátával.
        # Az 1..100. lépésben az lr fokozatosan nő. Ezután a futás tervezett
        # végéig koszinuszgörbén csökken a LearningRate 10%-ára.
        # Egy 20 lépéses új bemutatófutás még végig a bevezető szakaszban marad.
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
#  5. RÉSZ: GRADIENSELLENŐRZÉS
#     Egy kis modellen numerikus közelítéssel ellenőrizzük a gradienst.
#     Közelítés: (L(w+eps) - L(w-eps)) / (2*eps).
#     A teszt a mért pontokon hasonlítja össze ezt a backward eredményével.
# ============================================================================
<#
  Egy előjel- vagy indexhiba miatt a backward hibás gradienst is adhat.
  Itt másik úton ellenőrizzük: egy kiválasztott súlyt kicsit növelünk,
  majd csökkentünk, és mindkét esetben újraszámoljuk a loss-t.
  A különbségből megbecsüljük a loss súly szerinti deriváltját.

  Súlycsoportonként négy kiválasztott elemet vizsgálunk egy fix kis
  modellen. A numerikus eredmény közelítés: függ az eps értékétől
  és a lebegőpontos kerekítéstől. Az egyezés ezeken a pontokon
  támogatja a backward helyességét, de nem teljes bizonyítás.

  A teszt akkor jelez egyezést, ha minden mért relatív eltérés 1e-4 alatt van.
  A gradiensellenőrzés önmagában a tanítás eredményességét nem méri.
#>

function Invoke-GradCheck {
    # Bemenet: nincs (egy pici, fix modellt épít magának). Kimenet: konzolra
    # súlycsoportonként a legnagyobb relatív eltérés, végül GRADIENS OK / HIBAS.
    # Minden csoportból 4 véletlen súlyt mér - a teljes ellenőrzés túl lassú lenne.
    $script:Config = @{ VocabSize = 11; BlockSize = 6; EmbedDim = 8; NumHeads = 2; NumLayers = 2 }
    $cfg = $script:Config
    $P = New-GptParams $cfg -Seed 7
    # nagyobb sulyok, hogy a nemlinearitasok is "eljenek"
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
#  6. RESZ: GENERALAS
# ============================================================================
<#
  Generáláskor a súlyok rögzítettek. A ciklus:
    1. Feldolgozzuk a prompt karaktereit, és pontszámokat kapunk
       a következő karakterhez.
    2. A pontszámokból a temperature figyelembevételével
       valószínűségeket számolunk.
    3. Kiválasztunk egy karaktert, és kiírjuk.
    4. Ezt a karaktert is feldolgoztatjuk a modellel, majd ismétlünk.

  A temperature pozitív szám. 1 alatt a nagyobb esélyek dominálnak;
  1 felett egyenletesebb az eloszlás. Az 1 a módosítatlan softmax.
  A top-k szűrés után csak a K legvalószínűbb karakter marad választható.
  TopK=0 esetén nincs szűrés; TopK=1 mindig az egyik legnagyobb
  pontszámú karaktert választja, így a mintavétel nem ad változatosságot.

  A KV-cache a korábban számolt kulcs- és értékvektorokat őrzi meg.
  A MathNet változat ugyanazon modell számításait gyorsítja.
#>

function Sample-FromProbs {
    # A karaktert a megadott valószínűségek szerint választjuk ki.
    # Például [0.2, 0.3, 0.5] esetén a három karakter esélye 20%, 30%, 50%.
    # Egy véletlen számmal választunk a halmozott valószínűségek intervallumai közül.
    # Top-k mellett a K legnagyobb esélyt tartjuk meg, és az összegükből sorsolunk.
    # Ez a megtartott karakterek esélyeinek újranormalizálásával egyenértékű.
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

# A KV-cache rétegenként megőrzi a korábbi pozíciók k és v vektorait.
# Azonos súlyok és változatlan korábbi pozíciók mellett ezeket nem kell
# minden új karakternél újraszámolni. Az új q továbbra is összehasonlításra
# kerül a tárolt k vektorokkal, és a tárolt v vektorokból számolunk összeget.
# A gyorsulás a modelltől és az ablak hosszától függ.
# Ha a cache betelik, a kód az utolsó megközelítőleg negyed ablakot
# megtartja, majd 0-tól számozott pozíciókkal újrafeldolgozza.
# Ezért az ablak korábbi részének szövege kikerül az elérhető kontextusból.
# A cache a futás közbeni állapot része; az ürítése nem módosítja a súlyokat.
# ============================================================================
#  GYORSÍTÁS: MathNet.Numerics, opcionális
#  A mátrixműveletekhez a lib/MathNet.Numerics.dll könyvtárat használja.
#  Ugyanazokat a súlyokat és modelllépéseket alkalmazza, mint a PowerShell-változat.
#  Ha a könyvtár nem tölthető be, vagy -NoFast kapcsolót adunk meg,
#  a PowerShell-változat fut. A lebegőpontos eredmények kismértékben eltérhetnek.
#  Első olvasáskor a Forward-Token függvényt kövesd.
# ============================================================================

$script:Fast = $false
$script:FastW = $null

function Initialize-Fast {
    # Bemenet: a betöltött súlyok. Kimenet: nincs; beállítja a $script:Fast kapcsolót
    # és a $script:FastW-t (a súlymátrixok MathNet-be "becsomagolva", másolás nélkül).
    # Ha a DLL nincs meg vagy -NoFast van, csendben visszaesik a tiszta PowerShell útra.
    param([hashtable]$P)
    $script:Fast = $false
    if ($NoFast) { return }
    $dll = Join-Path $PSScriptRoot 'lib/MathNet.Numerics.dll'
    if (-not (Test-Path $dll)) { return }
    try { Add-Type -Path $dll -ErrorAction Stop } catch { return }
    $cfg = $script:Config; $d = [int]$cfg.EmbedDim; $ff = 4 * $d; $vocab = [int]$cfg.VocabSize
    $DM = [MathNet.Numerics.LinearAlgebra.Double.DenseMatrix]
    # A sor-folytonos W (in x out) pontosan a W^T (out x in) oszlop-folytonos tarolasa:
    # masolas nelkul lesz belole matrix, es y = W^T-matrix * a  ==  a * W.
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
    <# KV-cache a gyors uthoz: fejenkent egy (block x hd) matrix K-nak es V-nek, plusz munkapufferek. #>
    $cfg = $script:Config; $d = [int]$cfg.EmbedDim; $nL = [int]$cfg.NumLayers; $nH = [int]$cfg.NumHeads
    $hd = [int]($d / $nH); $ff = 4 * $d; $vocab = [int]$cfg.VocabSize; $block = [int]$cfg.BlockSize
    $DM = [MathNet.Numerics.LinearAlgebra.Double.DenseMatrix]; $DV = [MathNet.Numerics.LinearAlgebra.Double.DenseVector]
    $Kc = [object[]]::new($nL * $nH); $Vc = [object[]]::new($nL * $nH)
    for ($i = 0; $i -lt $Kc.Length; $i++) { $Kc[$i] = $DM::new($block, $hd); $Vc[$i] = $DM::new($block, $hd) }
    # A DenseVector::new(double[]) NEM masol: a vektor ugyanazt a tombot hasznalja, igy a
    # Multiply(v, eredmeny) kozvetlenul a mi double[]-unkba ir.
    $buf = @{}
    foreach ($spec in @(@('a', $d), @('q', $d), @('k', $d), @('v', $d), @('o', $d), @('proj', $d), @('h', $ff), @('m', $d),
                        @('logits', $vocab), @('scores', $block), @('p', $block), @('qh', $hd), @('kh', $hd), @('vh', $hd), @('oh', $hd))) {
        $arr = [double[]]::new($spec[1]); $buf[$spec[0]] = $arr; $buf[$spec[0] + 'V'] = $DV::new($arr)
    }
    return @{ K = $Kc; V = $Vc; Buf = $buf; Len = 0; Ids = [System.Collections.Generic.List[int]]::new(); LastAtt = [object[]]::new($nL * $nH); Fast = $true }
}

function Forward-TokenFast {
    # A Forward-Token MathNet-es ikertestvére: ugyanaz a lépéssor (embedding ->
    # rétegenként LN1, attention, LN2, MLP -> végső LN, Head), de a mátrixszorzásokat
    # a .NET könyvtár végzi. Bemenet: súlyok, KV-cache, egy karakter-id.
    # Kimenet: vocab darab logit (pontszám a következő karakterre).
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
        # LN1 (PowerShell-ciklus, 192 elem: olcso)
        $mean = 0.0; for ($j = 0; $j -lt $d; $j++) { $mean += $x[$j] }; $mean /= $d
        $var = 0.0; for ($j = 0; $j -lt $d; $j++) { $dev = $x[$j] - $mean; $var += $dev * $dev }
        $rs = 1.0 / [Math]::Sqrt($var / $d + 1e-5)
        for ($j = 0; $j -lt $d; $j++) { $a[$j] = ($x[$j] - $mean) * $rs * $g1[$j] + $b1[$j] }
        # q, k, v: harom matrix-vektor szorzas MathNet-tel
        $FW["Wq_$l"].Multiply($B.aV, $B.qV); $FW["Wk_$l"].Multiply($B.aV, $B.kV); $FW["Wv_$l"].Multiply($B.aV, $B.vV)
        # attention fejenkent: K_h * q_h -> pontszamok; softmax; V_h^T * p -> kimenet
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
    # Üres KV-cache létrehozása egy új beszélgetéshez / generáláshoz. Rétegenként
    # egy K és egy V tömb (block x d), plusz: Len (hány karakter van benne),
    # Ids (melyek), LastAtt (az utolsó karakter figyelmi térképe a /step kiíráshoz).
    param([hashtable]$P)
    if ($script:Fast) { return New-FastCache }
    $cfg = $script:Config; $d = [int]$cfg.EmbedDim; $n = [int]$cfg.BlockSize * $d
    $K = [object[]]::new($cfg.NumLayers); $V = [object[]]::new($cfg.NumLayers)
    for ($l = 0; $l -lt $cfg.NumLayers; $l++) { $K[$l] = [double[]]::new($n); $V[$l] = [double[]]::new($n) }
    return @{ K = $K; V = $V; Len = 0; Ids = [System.Collections.Generic.List[int]]::new(); LastAtt = [object[]]::new($cfg.NumLayers * $cfg.NumHeads) }
}

function Forward-Token {
    <#
  Feldolgoz egy karakterazonosítót, és visszaadja a következő karakter
  pontszámait: egy logit jut a szókészlet minden elemére.
  Közben frissíti a KV-cache-t a feldolgozott karakter adataival.
  Az eloszlássá alakítás és a karakterválasztás később történik.
  Ez csak forward számítás: a súlyokat nem módosítja.
#>
    param([hashtable]$P, [hashtable]$Cache, [int]$Id)
    if ($script:Fast) { return ,(Forward-TokenFast $P $Cache $Id) }
    $cfg = $script:Config
    $d = [int]$cfg.EmbedDim; $nL = [int]$cfg.NumLayers; $nH = [int]$cfg.NumHeads
    $hd = [int]($d / $nH); $ff = 4 * $d; $vocab = [int]$cfg.VocabSize; $block = [int]$cfg.BlockSize
    $attScale = 1.0 / [Math]::Sqrt([double]$hd); $geluC = [Math]::Sqrt(2.0 / [Math]::PI)

    # Betelt az ablak: az utolsó megközelítőleg negyedét újrafeldolgozzuk 0-tól számozott pozíciókkal.
    if ($Cache.Len -ge $block) {
        $keepFrom = $block - [int]($block / 4)   # az utolso negyedet tartjuk meg (ritkabb ujraepites)
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
        # q, k, v az uj poziciora (vektor x matrix)
        $q = [double[]]::new($d); $kb = $curPos * $d
        for ($i = 0; $i -lt $d; $i++) {
            $ai = $a[$i]; if ($ai -eq 0.0) { continue }; $wi = $i * $d
            for ($j = 0; $j -lt $d; $j++) { $q[$j] += $ai * $Wq[$wi + $j]; $Kc[$kb + $j] += $ai * $Wk[$wi + $j]; $Vc[$kb + $j] += $ai * $Wv[$wi + $j] }
        }
        # attention: az uj karakter figyel 0..pos-ra
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
            $Cache.LastAtt[$l * $nH + $h] = $scores   # vizualizaciohoz: kire figyelt az uj karakter
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
    # vegso LN + head
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
    # Bemenet: nyers logit-ek. Kimenet: egy húzott karakter-id.
    # A logit-eket elosztjuk a temperature-rel (ez "élesíti" vagy "laposítja" az
    # eloszlást), softmax-szal valószínűséggé alakítjuk, majd Sample-FromProbs.
    param([double[]]$Logits)
    $Temperature = $script:Temperature; $TopK = $script:TopK
    $max = -1e300; foreach ($l in $Logits) { if ($l -gt $max) { $max = $l } }
    $probs = [double[]]::new($Logits.Length); $sum = 0.0
    for ($c = 0; $c -lt $Logits.Length; $c++) { $probs[$c] = [Math]::Exp(($Logits[$c] - $max) / $Temperature); $sum += $probs[$c] }
    for ($c = 0; $c -lt $Logits.Length; $c++) { $probs[$c] /= $sum }
    return Sample-FromProbs $probs $TopK
}

function Format-TraceChar {
    <# Lathatova teszi a nem-nyomtathato karaktereket a nyomkovetesben. #>
    param([string]$ch)
    switch ($ch) { "`n" { return '\n' } "`r" { return '\r' } "`t" { return '\t' } ' ' { return [char]0xB7 } default { return $ch } }
}

function Get-NextStepTrace {
    <#
  A /step nézethez szöveges összefoglalót és egy kiválasztott karaktert ad.
  A már kiszámolt logits értékeit használja, ezekből a megadott temperature
  mellett valószínűségeket számol, majd a TopK szerint mintát vesz.

  Megmutatja a kontextus hosszát, rétegenként a fejek átlagában
  legnagyobb attention-súlyú pozíciót, valamint legfeljebb öt jelöltet.
  Ez néhány köztes eredmény összefoglalója, nem a teljes számítás naplója.
  Az attention-súly önmagában nem magyarázza meg a kimenet kiválasztását.

  Kimenet: @{ Lines; NextId; NextChar }.
  A cache-t nem módosítja. A hívó dolgozza fel a kiválasztott karaktert.
#>
    param([hashtable]$P, [hashtable]$Cache, [double[]]$Logits, [double]$Temp, [int]$TopK, [string]$Lang = 'hu')
    $cfg = $script:Config
    $nL = [int]$cfg.NumLayers; $nH = [int]$cfg.NumHeads; $vocab = $Logits.Length
    $curPos = $Cache.Len - 1
    $en = ($Lang -eq 'en')
    $esc = [char]27; $dim = "$esc[90m"; $cy = "$esc[36m"; $wh = "$esc[97m"; $or = "$esc[38;5;208m"; $rst = "$esc[0m"; $bold = "$esc[1m"; $grn = "$esc[32m"

    # --- softmax homersekletlel: logit -> valoszinuseg ---
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
    # retegenkent: a fejek atlagolt attention-je alapjan a legerosebb pozicio
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
    <# Betaplalja a Seed szoveget, majd Count karaktert general (streamelve).
       Ez a legegyszerűbb generáló ciklus: Seed karakterei -> cache, majd
       Count-szor: húzunk egy karaktert, kiírjuk, visszatápláljuk. A szótárban
       nem szereplő karaktereket (amiket a modell sosem látott) egyszerűen kihagyja. #>
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
  Soronként működő szövegfolytató felület, átirányított be- és kimenethez is.
  A beírt sor és egy újsor a futás közbeni kontextushoz kerül.
  A modell ezt folytatja; a tanítószöveget és a súlyokat nem módosítjuk.

  Legfeljebb MaxTokens új karaktert ír. Egymást követő két újsornál
  korábban is leállhat, ha már elérte a hosszkorlát felét.
  A /reset törli a kontextust. A /step 1 egy karakter kiválasztását mutatja meg.
  További parancsok: /temp 0.8, /tokens 300, /topk 10, /help, /q.
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

    if ($en) { try { [Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::InvariantCulture } catch { } }   # '.' tizedes, vesszos ezres
    try { [Console]::OutputEncoding = [Text.Encoding]::UTF8; [Console]::InputEncoding = [Text.Encoding]::UTF8 } catch { }
    try { Clear-Host } catch { }
    $title = "nanoGPT-ps  " + [char]0xB7 + "  $modelName"
    $info = if ($en) { ("{0} layers " + [char]0xB7 + " dim {1} " + [char]0xB7 + " block {2} " + [char]0xB7 + " step {3} " + [char]0xB7 + " {4:N0} params") -f $cfg.NumLayers, $cfg.EmbedDim, $cfg.BlockSize, $Step, ($P.Values | ForEach-Object { $_.Length } | Measure-Object -Sum).Sum }
             else    { ("{0} reteg " + [char]0xB7 + " dim {1} " + [char]0xB7 + " blokk {2} " + [char]0xB7 + " {3}. lepes " + [char]0xB7 + " {4:N0} parameter") -f $cfg.NumLayers, $cfg.EmbedDim, $cfg.BlockSize, $Step, ($P.Values | ForEach-Object { $_.Length } | Measure-Object -Sum).Sum }
    # Fejléc: a pixelrajz portré (4 sor) balra, a három szövegsor jobbra, lekerekített dobozban.
    # A logósorok ANSI színkódokat tartalmaznak, ezért a .Length nem a látható szélesség; az a szélesség
    # rögzített 12 oszlop (a Get-PixelLogo 12 oszlop széles sorokat ad). A doboz igazítását a látható szélességből számoljuk.
    $kind = if ($isSim) { 'orban' } else { 'shakespeare' }
    $logo = Get-PixelLogo $kind
    $logoW = 12
    $simText = if ($isSim) { if ($en) { 'GENERATED TEXT - simulation, not a real quote' } else { 'GENERALT SZOVEG - szimulacio, nem valodi idezet' } } else { '' }
    $textRows = @($title, $info, $simText, '')
    $maxText = ($textRows | ForEach-Object { $_.Length } | Measure-Object -Maximum).Maximum
    $inner = 2 + $logoW + 3 + $maxText + 2   # bal margó + logó + rés + legszélesebb szöveg + jobb margó
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

    $logits = $null   # a legutobbi tokenbol szuletett josolat; a fordulok kozott is megorizzuk (/step)
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

        # --- a felhasznalo sora bemegy a kontextusba ---
        $logits = $null
        foreach ($ch in ($line + "`n").ToCharArray()) {
            $s = [string]$ch
            if ($script:CharToId.ContainsKey($s)) { $logits = Forward-Token $P $cache $script:CharToId[$s] }
        }
        if ($null -eq $logits) { $logits = Forward-Token $P $cache 0 }

        # --- valasz streamelve ---
        [Console]::Write("$orange$bold" + [char]0x25C6 + " $reset$white")
        $sw = [Diagnostics.Stopwatch]::StartNew(); $n = 0; $lastCh = ''
        for ($i = 0; $i -lt $maxTok; $i++) {
            $script:Temperature = $temp; $script:TopK = $topk
            $next = Sample-Logits $logits
            $ch = $script:IdToChar[$next]
            if ($ch -eq "`n" -and $lastCh -eq "`n" -and $n -ge [int]($maxTok / 2)) { break }   # bekezdes vege = valasz vege
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
      Pixel-art fej a modellhez, Claude Code stilusban. 12 x 8 pixel, fel-blokk
      karakterekkel (egy szoveg-sor = 2 pixel-sor), 256 szinnel.
      Visszaad 4 db szoveg-sort.
      (Tisztán dekoráció a teljes képernyős chathez; a modellhez semmi köze.)
    #>
    param([string]$Kind)
    # palettak: betu -> 256-szin
    if ($Kind -eq 'orban') {
        # osz rovid haj, kerek arc, sotet oltony, feher ing, piros nyakkendo
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
  A szövegfolytató felület teljes képernyős változata.
  Az Invoke-ChatPlain logikájához képernyőrajzolás, fejléc és állapotsor társul.
  A Wrap, Render és Read-Input segédfüggvények a megjelenítést kezelik;
  a modell működésének megértéséhez elsőre átugorhatók.
  A -Chat belépési pont az oktatáshoz az Invoke-ChatPlain felületet használja.
  Parancsok: /temp 0.8, /tokens 300, /topk 10, /step 1, /reset, /help, /q.
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

    # a beszelgetes sorai (mar tordelve)
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
        # fejlec
        for ($i = 0; $i -lt 4; $i++) {
            $txt = switch ($i) { 0 { $title } 1 { $line2 } 2 { $line3 } default { '' } }
            [Console]::SetCursorPosition(0, $i); [Console]::Write('  ' + $logo[$i] + '   ' + $txt)
        }
        # beszelgetes: az also resz 4 sor (jobb-status, vonal, input, vonal, statusz = 5)
        $bodyTop = $headerRows; $bodyRows = $h - $headerRows - 5
        $start = [Math]::Max(0, $lines.Count - $bodyRows)
        for ($i = 0; $i -lt $bodyRows -and ($start + $i) -lt $lines.Count; $i++) {
            [Console]::SetCursorPosition(0, $bodyTop + $i)
            $l = $lines[$start + $i]; if ($l.Length -gt $w - 1) { $l = $l.Substring(0, $w - 1) }
            [Console]::Write($l)
        }
        # also blokk
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

    $logits = $null   # a legutobbi josolat; a fordulok kozott is megorizzuk (/step)
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

        # a felhasznalo sora bemegy a kontextusba
        $logits = $null
        foreach ($ch in ($line + "`n").ToCharArray()) {
            $s = [string]$ch
            if ($script:CharToId.ContainsKey($s)) { $logits = Forward-Token $P $cache $script:CharToId[$s] }
        }
        if ($null -eq $logits) { $logits = Forward-Token $P $cache 0 }

        # streamelt valasz: a lines vegere irunk, es minden ujsornal / sorhossz-tullepesnel ujrarajzolunk
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
#  7. RESZ: FOPROGRAM
# ============================================================================
<#
  Itt választjuk ki a futási módot, ebben a sorrendben:
    -AsLibrary: a definíciók és a kezdeti állapot betöltése után visszatér.
    -GradCheck: gradiensellenőrzést futtat egy kis modellen.
    -Train: elindítja vagy folytatja a tanítást.
    Egyébként: betölti a súlyokat, és szöveget generál.
  A -Chat a generáláshoz interaktív felületet nyit.
  A -Prompt a közvetlen generálás kezdőszövege.
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

if ($AsLibrary) { return }   # dot-source-olva csak a fuggvenyeket toltjuk be (viz.ps1 hasznalja)
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
    # A görgethető felületen a /step hosszabb kimenete is visszaolvasható.
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
