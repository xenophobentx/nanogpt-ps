# Nézd meg, hogyan folytat szöveget egy GPT

Ebben a modellben egy token egy karakter. A modell minden lépésben
pontszámot ad minden ismert karakternek. Ezekből esélyeket számolunk,
kiválasztunk egy karaktert, és hozzáírjuk az eddigi szöveghez.

A futtatáshoz PowerShell 7 kell. A parancsokat a `nanogpt-ps.ps1`
mappájából add ki. Ha a repó gyökerében állsz, előbb lépj át:
`Set-Location .\hu`.

## 1. Generálj rövid szöveget

```powershell
pwsh -File .\nanogpt-ps.ps1 -Lang hu -Prompt "ROMEO:" -MaxTokens 40
```

A `ROMEO:` a kezdőszöveg, a `MaxTokens 40` legfeljebb 40 új karaktert kér.
A program a saját mappájában lévő `nanogpt-shakespeare-weights.json` fájlt tölti be.
A `-Lang hu` a felület nyelve. A Shakespeare-szövegen tanult modell ettől
még angol szöveget folytat.

A kimenet futásonként eltérhet, mert a karaktereket az esélyek szerint
választjuk. Ilyen kis modellnél a hosszabb szöveg gyakran szétesik.

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
A `/step 1` az utána következő karakter kiválasztását mutatja meg.
A kontextusban ott van a beírt sor, egy hozzáadott újsor és a már legenerált karakter.

Figyeld meg a jelöltek esélyét és a kiválasztott karaktert. Nem mindig
a legvalószínűbb karakter nyer. A kiválasztott karakter bekerül a
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

A logitok pontszámok, a softmax ezekből csinál valószínűségeket.
Az attention-súlyok azt szabályozzák, milyen arányban keveredjenek a pozíciók
értékvektorai. Ezek nem azonosak a következő karakter esélyeivel. A `/step`
csak a fontosabb eredményeket mutatja, nem minden köztes tömböt.

## 3. Három rövid kísérlet

### Temperature: ugyanaz a bemenet, más eloszlás

A chatben:

```text
/reset
/topk 0
/temp 0.3
/step 1
/reset
/temp 1.2
/step 1
```

Üres kontextusnál a `/step` újsorral indul, ha a modell ismeri az újsort,
ha nem, akkor a 0-s karakterazonosítóval. A két rész így azonos bemenetről indul.
Hasonlítsd össze az esélyeket: alacsonyabb temperature-nél a nagy
valószínűségek még jobban elnyomják a kicsiket. Ha ugyanaz a karakter nyer is,
az eloszláson látszik a különbség.

### Top-k: hány jelölt maradhat?

A chatben:

```text
/reset
/temp 0.8
/topk 1
/step 1
```

Egyetlen jelölt marad. Szűrés után 100% eséllyel ez nyer, akkor is,
ha előtte kisebb szám állt mellette. Ha megismétled a `/reset` és a
`/step 1` parancsot, mindig ugyanazt kapod: azonos kezdőállapotból, azonos
beállításokkal a mintavétel nem hoz változatosságot.

Példa: a=50%, b=30%, c=20%. Top-k=2 után c kiesik; a esélye 50/80=62,5%,
b esélye 30/80=37,5% lesz. A top-5 csak azt szabja meg,
hány sort látsz, a top-k pedig azt, hány jelölt közül választ a modell.

### Kontextus: számít-e az előzmény?

Ugyanebben a chatben:

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

Mindkét `/step` azt az állapotot mutatja, amikor a program már automatikusan
kiírt egy karaktert. A jelöltlisták eltérhetnek, mert más az előzmény. Egy ilyen
rövid kísérletből nem lehet arra következtetni, hogy valamelyik attention-fejnek
rögzített nyelvtani szerepe van.

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
Ha 50%-ot adott, a loss körülbelül 0,693. Minél nagyobb esélyt ad a helyes
folytatásnak, annál kisebb a loss.

A backward kiszámolja, hogyan változna a loss a súlyok kis módosítására.
Az Adam ebből és az előző lépések mozgóátlagaiból számolja ki, mennyit
módosítson a súlyokon. Generálás közben se backward, se súlyfrissítés nincs.

## 5. Kis tanítási bemutató

```powershell
pwsh -File .\nanogpt-ps.ps1 -Lang hu -Train -Layers 1 -Dim 16 -Heads 2 -Block 16 -BatchSize 1 -Threads 1 -TrainSteps 20 -CorpusFile .\tinyshakespeare.txt -WeightsFile .\demo-weights.json
```

Ha a `demo-weights.json` még nincs meg, a program új modellt csinál. Ha megvan,
onnan folytatja, a fájlban tárolt modellmérettel. Ha tiszta lappal akarsz
indulni, adj meg másik fájlnevet.
A `tinyshakespeare.txt` fájlnak a mappában kell lennie.

A 20 lépés csak arra jó, hogy lásd, hogyan működik. Ennyi lépés után még
a tanulási ráta felfutásánál tartunk, olvasható szöveget ne várj. A loss
ingadozhat, mert mindig más szövegablakon mérjük. Attól, hogy a tanítási
loss csökken, még nem tudod, mit kezd a modell új szöveggel.

A kézzel írt backwardot külön is ellenőrizheted:

```powershell
pwsh -File .\nanogpt-ps.ps1 -Lang hu -GradCheck
```

Ez egy kis modellen néhány kiválasztott gradienst összevet a numerikus
közelítéssel. A szöveg minőségéről nem mond semmit.

## 6. Innen olvasd a kódot

Keress ezekre a nevekre, ebben a sorrendben:

1. `Invoke-Generate`: a karakterválasztás ismétlése.
2. `Forward-Token`: egy karakter feldolgozása.
3. `Sample-Logits`, `Sample-FromProbs`: pontszámokból karakterválasztás.
4. `Compute-SeqGrad`, FORWARD: tanítási veszteség számítása.
5. `Compute-SeqGrad`, BACKWARD: gradiensek számítása.
6. `Invoke-Train`: súlyfrissítés.

A képernyőrajzolás, a mentés és a MathNet-gyorsítás ráér később.
