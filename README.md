# KI_ML_with_Python
AI and machine learning with Python. Course from NITO 27. november and 01. december.

Kursholder e-post: `kristian@botnen.org`

## Day 1 (27. november)

ChatGPT er ein modell som skal generere tekst basert på innputt. Ein spesifikk modell
Kan også ha modeller for å spå været, eller teikne bilder.

Kvar gjeng grensa mellom automatisering og KI? Ikkje noke godt svar på det.
Begge har sitt bruksområde. 

Kva er mulig (teknologi)? Kva er greit (etikk)? Hva er lov (juss)?
Kan være eit bias i svara oss får frå ein KI. Er vanskelig å finne ut av, anna enn å være
oppmerksom på svaret ein får. Kan for eksempel være farga av kordan ting var før, men som ikkje
er gjeldande i dag (som lønnsforskjeller).

KI er verken kunstig eller intelligent. Må være noke som bruker den og mater den med innputt.

### Maskinlæring 
Mål å trene modellen med kjent innputt, slik at det kjem noke fornuftig ut når den får ny 
innputt seinare.

Modellen får data frå eit treningssett. Justerer parameterne slik at det som kjem ut blir det oss 
ønsker. Gir modellen data som oss veit er viktig.

__Framgangsmåte for å trene ein modell:__

1. Treningsdatasett
2. Velge kva modell oss ønsker å trene
   - Kjem an på kva oss vil løse
3. Tren modellen og gi den inndata
4. Gjenta steg 3 til oss er fornøgd
5. Bruke modellen på nye ukjente data

Dette som er veilledet læring innen ML.
Kan gi sannsynlighet som utputt (f.eks. 0.02 for at det er positivt, 0.98 for at noke er negativt).

Nokre biblioteker for å jobbe med slikt:

- PyTorch
- TensorFlow
  - Ser ut som at den her skal oss ikkje bruke
- OpenCV
  - CV stend for Computer Vision
- scikit learn

Koden i scikit_ex.py blir kjørt i Kaggle, er ein del 
ting som må innstalleres dersom ein skal kjøre det på maskina.

Vanlig å dele datasetta sine i to, slik at ein har ein del ein trener med og ein del ein tester med.
SciKit Learn har ein metode for å gjere det her. Linje 61 i scikit_ex.py.

`fit` er eit navn ofte brukt for å trene modellen.
`y_train` er fasiten, linje 60 ish.

__Framgangsmåte:__
Ein har eit datasett. Ein må sette seg inn i det, dele det i to og deretter 
trene modellen din på settet.

FastAI har ein måte å detektere om bildet blei korrekt lasta ned.

I FastAI er det DataBlock som deler opp datasettet i trening og verifisering.
Kan skalere bildane ned for å spare tid. Når det er gjort har oss eit sett med treningsdata basert på
bilda som vi lasta ned.
Deretter kan vi trene modellen vår. I sted brukte vi nærmeste nabo, no bruker vi Resnet18, 
brukes på bildebehandling.
(Nest siste blokka): Trener no ein AI modell på bilder henta frå internett. 

Har forskjellige Resnet-modeller. Tallet bak sier noke om kor mange lag den er blitt trent på.
Desto høyere tallet er desto lavere er feilraten(?).
Bruker 18 her for at det er litt raskere(?).

_Pause_

Koden til foredragsholder ligger i prosjektet `pythonki-main.zip`.

Laster inn kjennetegn på eit ansikt, justere bildet ned til gråtoner, scanne bildet, skriv resultatet
over på bildet (firkanter rundt ansikter f.eks.), og deretter vis resutlatet.

Oppgave til fredag, gjenkjenne ansikter ansikter i en videostrøm! 
Høres ut som at det same oss gjorde i det første kurset.


__Endex.__

## Day 2 (01. december)

