[In English: README.md](README.md)

# psgpt

Karakterszintű GPT, tisztán PowerShellben a nanoGPT alapján. A transformer benne van a szkriptben. Van hozzá terminálos chat is.

Azért érdekes, mert az egészet el lehet olvasni. Minden mátrixszorzás egy ciklus, amire be breakelhetsz. 

És a chatben a `/step használatával követni lehet következő karakter kiválasztás folyamatát`.

![A nanoGPT-ps chatablak: a modell fejléce, majd egy Shakespeare-stílusú folytatás](docs/chat.png)

*A chatablak: a modell fejléce, majd egy Shakespeare-stílusú folytatás.*

## Mire jó

Beszélgetni. Beírsz egy sort, a modell folytatja abban a stílusban, amin tanult:

```
  > ROMEO:
  ◆ KING RICHARD II:
    And thou consent is so this first wivers,
    For thou art thou only to cheek the corruption,
```

Tanítás: A `nanogpt-ps.ps1 -Train` 

## Gyors indítás

A súlyfájlok darabja kb. 57 MB,  a GitHub Releases alatt megtalálható. Töltsd le a `nanogpt-shakespeare-weights.json` és a `nanogpt-orban-weights.json` fájlt abba a mappába ahol a ps scriptek vannak.

```
cd hu          # vagy: cd en
# a két súly .json a Releases-ből ide, ebbe a mappába
pwsh ./chat.ps1 -Model shakespeare      # vagy: -Model orban
```

Írj be valamit, Enter. A chaten belül:

- `/step` vagy `/step 5`: karakterenkénti generálás a teljes bontással
- `/temp`, `/tokens`, `/topk`: mintavételi hőmérséklet, kimenet hossza, top-k levágás
- `/help`: a többi
- `/q`: kilépés

## A két modell

Mindkettő kb. 2,7 millió paraméteres, 128 karakteres kontextusablakkal.

A `shakespeare` a Tiny Shakespeare szövegen tanult, ez nagyjából 1 MB a drámákból. Angolul ír, színdarab formában: csupa nagybetűs szereplőnevek, sortörés ott, ahol a szöveg kívánja, a szavak többnyire valódiak, néha csak majdnem.

Az `orban` magyar politikai beszédeken tanult. Amit kiad, az stílusszimuláció: kitalált mondatok a forrás ritmusával és szókincsével. 

## Sebesség és határok

Lassú. Karakterszintű, interpretált, PowerShell: laptopon nagyjából 36 karakter másodpercenként, gyorsabb asztali gépen a natív könyvtárral 84 körül.  A mellékelt méretű modell betanítása CPU-n többnapos meló.

Kicsi 2,7 millió paraméter és 128 karakternyi kontextus arra elég, hogy megtanulja a helyesírást, a központozást meg egy szöveg lüktetését, sokkal többre nem. Kérdésekre nem válaszol. Tekintsd egy GPT-nek, amit elejétől végéig el lehet olvasni; a generált szöveg csak bizonyíték arra, hogy amit olvastál, működik.

## Követelmények

PowerShell 7, Windowson, Linuxon vagy macOS-en. Se Python, se ML-keretrendszer, se GPU.

A `lib/MathNet.Numerics.dll` a szkriptek mellett van, és nem kötelező. Ha megvan, a generálás kb. 8-szor gyorsabb. Nélküle minden ugyanúgy megy a tiszta PowerShell-ágon.

## en/ és hu/

A repóban két példány van a kódból: az `en/` angol, a `hu/` magyar kommentekkel. A kód ugyanaz; azt a mappát válaszd, amelyiknek a kommentjeit szívesebben olvasod, és oda tedd a súlyfájlokat is.

## Licenc

MIT. A LICENSE fájl itt van a README mellett.
