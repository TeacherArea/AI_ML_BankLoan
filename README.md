# Grundläggande beslutsträd för lånebedömning

Detta projekt demonstrerar hur en enkel maskininlärningsmodell kan byggas med Python och `scikit-learn`. Exemplet används i undervisning Artificeill Intelligens 1 på gymnasienivå för att introducera elever till grundläggande koncept inom

- datastrukturer och tabeller (pandas)
- numeriska simuleringar (numpy)
- maskininlärning
- tränings- och testdata
- klassificering
- visualisering av beslutsträd
- utvärdering av modeller

Syftet är inte att skapa en realistisk kreditmodell, utan att stegvis utforska hur kod kan användas för att beskriva och analysera data.


## Vad modellen gör

Programmet genererar syntetisk data för 500 personer:

- årsinkomst
- ålder
- kreditvärdighet

Utifrån dessa variabler skapas ett enkelt beslut om ett lån ska beviljas eller avslås.

Beslutet baseras egentligen på två faktorer:

- inkomsten måste vara tillräckligt hög
- kreditvärdigheten måste vara över en viss nivå

Lite slumpmässigt brus läggs också till för att simulera att verklig data ofta innehåller variationer och osäkerhet.


## Några pedagogiska noteringar

Variabeln age används inte när lånebeslutet genereras. Likaså heller inte print(data) eller 

Detta är avsiktligt, då syftet är att elever själva ska upptäcka detta och undersöka

- vilka variabler modellen faktiskt använder
- hur irrelevanta variabler kan finnas i data, och vad som kan hända när de blir aktiva
- hur ett beslutsträd väljer de mest informativa egenskaperna
- varför vissa variabler ignoreras trots att de finns med i datasetet

Detta öppnar för diskussioner om:

- modelltolkning
- relevans
- bias
- feature selection
- korrelation kontra kausalitet


## Modell

Modellen som används är ett beslutsträd (DecisionTreeClassifier) med initialt max_depth = 3


## Använda bibliotek
pandas numpy scikit-learn matplotlib, för export joblib (senare)

För att använda: pip install pandas numpy scikit-learn matplotlib. Uppdatera eventuellt (python.exe -m pip install --upgrade pip).