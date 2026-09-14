# Nézd meg, hogyan folytat szöveget egy GPT

Ebben a modellben egy token egy karakter. A modell minden lépésben
pontszámot ad az ismert karaktereknek. Ezekből esélyeket számolunk,
kiválasztunk egy karaktert, és hozzáírjuk az eddigi szöveghez.

A futtatáshoz PowerShell 7 kell. A parancsokat a `nanogpt-ps.ps1`
mappájából add ki. A teljes repository gyökeréből először:
`Set-Location .\dist\hu`. Önálló csomagnál annak saját mappájából indulj.

## 1. Generálj rövid szöveget

```powershell
pwsh -File .\nanogpt-ps.ps1 -Lang hu -Prompt "ROMEO:" -MaxTokens 40
```

A `ROMEO:` a kezdőszöveg, a `MaxTokens 40` legfeljebb 40 új karaktert kér.
A program a saját mappájában levő `nanogpt-shakespeare-weights.json` fájlt tölti be.
A `-Lang hu` a felület nyelve. A Shakespeare-szövegen tanult modell ettől
még angol szöveget folytat.

A kimenet futásonként eltérhet, mert a karaktereket az esélyek szerint
választjuk. Ilyen kis modellen a hosszabb szöveg gyakran elveszíti az összefüggést.

## 2. Nézz meg egyetlen karakterválasztást

```powershell
pwsh -File .\nanogpt-ps.ps1 -Lang hu -Chat -MaxTokens 1
```

A felületen írd be:

```text
ROMEO:
/step 1
```

A `ROMEO:` beírása után a program már generál egy karaktert.
A `/step 1` az azt követő karakter választását mutatja meg.
A kontextus a beírt sort, a hozzáadott újsort és a generált karaktert is tartalmazza.

Figyeld meg a jelöltek esélyét és a kiválasztott karaktert. Nem kell mindig
a legvalószínűbb karakternek nyernie. A kiválasztott karakter bekerül a
kontextusba, így a következő `/step 1` már más bemenetből indul. Kilépés: `/q`.

```text
szöveg -> karakterazonosítók -> karakter- és pozícióvektorok
       -> transformer-rétegek -> végső LayerNorm -> Head -> logits
       -> temperature + softmax -> top-k -> karakterválasztás
                                                   |
                         következő bemenet <-------+

Egy transformer-réteg:
  LayerNorm -> attention -> bemenet hozzáadása
  LayerNorm -> MLP       -> bemenet hozzáadása
```

A logits pontszámokat jelent; a softmax ezekből készít valószínűségeket.
Az attention-súlyok a pozíciók értékvektorainak keverését szabályozzák.
Ezek különböznek a következő karakter esélyeitől. A `/step` néhány eredményt
mutat meg, nem minden köztes számtömböt.

## 3. Három rövid kísérlet

**Temperature: ugyanaz a bemenet, más eloszlás.** A chatben:

```text
/reset
/topk 0
/temp 0.3
/step 1
/reset
/temp 1.2
/step 1
```

Üres kontextusnál a `/step` újsorral indul, ha a modell ismeri azt;
különben a 0-s karakterazonosítóval. A két rész így azonos bemenetről indul.
Hasonlítsd össze az esélyeket: alacsonyabb temperature mellett a nagyobb
valószínűségek jobban dominálnak. Nem kell más karakternek nyernie ahhoz,
hogy az eloszlás változása látható legyen.

**Top-k: hány jelölt maradhat?** A chatben:

```text
/reset
/temp 0.8
/topk 1
/step 1
```

Egyetlen jelölt marad. A szűrés utáni választási esélye 100%, akkor is,
ha a szűrés előtti kijelzésben kisebb szám szerepelt. Ismételd meg a
`/reset`, majd `/step 1` parancsot: azonos kezdőállapotból és beállításokkal
a mintavétel nem ad változatosságot.

Példa: a=50%, b=30%, c=20%. Top-k=2 után c kiesik; a esélye 50/80=62,5%,
b esélye 30/80=37,5% lesz. A top-5 a kijelzett sorok száma;
a top-k a választásban részt vevő jelöltek száma.

**Kontextus: számít-e az előzmény?** Ugyanebben a chatben:

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

Mindkét `/step` az automatikusan kiírt egy karakter utáni állapotot mutatja.
A jelöltlisták eltérhetnek az eltérő szövegelőzmény miatt. Egy rövid
kísérletből nem következik, hogy egy attention-fejnek rögzített nyelvtani szerepe van.

## 4. Hogyan lesz a szövegből tanítófeladat?

```text
Szöveg:    a l m a
Bemenet:   a l m
Cél:       l m a

Látható szöveg -> elvárt következő karakter
             a -> l
            al -> m
           alm -> a
```

Tanításkor a valódi folytatást ismerjük. Ha az első pozícióban a modell
az `l` karakternek 10% esélyt adott, a loss ott `-ln(0.1)`, körülbelül 2,303.
Ha 50%-ot adott, a loss körülbelül 0,693. A helyes folytatás nagyobb esélye
tehát kisebb veszteséget jelent.

A backward kiszámolja, hogyan változna a loss a súlyok kis módosítására.
Az Adam ebből és az előző lépések mozgóátlagaiból súlyfrissítést számol.
Generálás közben nincs backward és súlyfrissítés.

## 5. Kis tanítási bemutató

```powershell
pwsh -File .\nanogpt-ps.ps1 -Lang hu -Train -Layers 1 -Dim 16 -Heads 2 -Block 16 -BatchSize 1 -Threads 1 -TrainSteps 20 -CorpusFile .\tinyshakespeare.txt -WeightsFile .\demo-weights.json
```

Ha a `demo-weights.json` még nem létezik, új modell készül. Létező fájlból
folytatja, annak modellméretével. Új kísérlethez adj másik súlyfájlnevet.
A `tinyshakespeare.txt` fájlnak a mappában kell lennie.

A 20 lépés a működés megfigyelésére szolgál. Még a tanulási ráta bevezető
szakaszában járunk; ne várj jól olvasható szöveget. A loss ingadozhat,
mert különböző ablakokon mérjük. A csökkenő tanítási loss önmagában nem
mutatja meg, hogyan teljesít a modell új szövegen.

A kézzel írt backward külön ellenőrzése:

```powershell
pwsh -File .\nanogpt-ps.ps1 -Lang hu -GradCheck
```

Ez kiválasztott gradienseket vet össze numerikus közelítéssel egy kis modellen.
Nem a modell szövegminőségét méri.

## 6. Innen olvasd a kódot

Keress ezekre a nevekre, ebben a sorrendben:

1. `Invoke-Generate`: a karakterválasztás ismétlése.
2. `Forward-Token`: egy karakter feldolgozása.
3. `Sample-Logits`, `Sample-FromProbs`: pontszámokból karakterválasztás.
4. `Compute-SeqGrad`, FORWARD: tanítási veszteség számítása.
5. `Compute-SeqGrad`, BACKWARD: gradiensek számítása.
6. `Invoke-Train`: súlyfrissítés.

A képernyőrajzolást, mentést és MathNet gyorsítást később is megnézheted.
