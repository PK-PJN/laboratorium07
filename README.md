# Przetwarzanie języka naturalnego
# Laboratorium 7: analiza wydźwięku

1. Skopiować zawartość niniejszego repozytorium
na dysk lokalny. Pobrać korpus opinii `dataset_clarin.zip`
z linku na dole strony https://clarin-pl.eu/dspace/handle/11321/700
i rozpakować go.

    ```
    git clone https://github.com/PK-PJN/laboratorium07.git
    cd laboratorium07
    unzip dataset_clarin.zip
    ```

2. Zainstalować `scikit-learn` — bibliotekę
z narzędziami do uczenia maszynowego.
Od razu zainstalują się również niezbędne
biblioteki `numpy` i `scipy`.

    ```
    pip install sklearn
    ```

3. Dzisiejsze zadanie dotyczy analizy
wydźwięku (*sentiment analysis*), czyli
klasyfikowania opinii jako pozytywne
lub negatywne zależnie od ich treści.
Zostanie ono rozwiązane za pomocą
*regresji logistycznej*. W bibliotece
`scikit-learn` odpowiada za nią klasa
[`linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

4. Skrypt `zadanie.py` działa następująco:

    * Wczytuje zbiór uczący
    (`dataset/*.train.txt`).

    * Zamienia znaki w każdym wierszu
    zbioru uczącego na małe.

    * Twórcy zbiorów danych z CLARIN-PL
    zadbali o tokenizację wierszy,
    więc się nią nie przejmujemy.

    * Skrypt dodaje do zmiennej `X_train`
    kolejne wiersze, a do zmiennej
    `Y_train` — wartości 0 (opinia
    negatywna) lub 1 (opinia pozytywna).

    * Buduje model regresji logistycznej
    na podstawie zmiennych `X_train`
    i `Y_train`.

    * Wyświetla parametry modelu.

    * Wyświetla 20 segmentów, które
    mają najbardziej dodatni wpływ
    na wynik regresji logistycznej,
    i 20 segmentów, które mają
    na niego najbardziej ujemny wpływ.

    * Wyświetla raport o jakości
    klasyfikacji zbioru testowego
    (`dataset/*.test.txt + dataset/*.dev.txt`)
    za pomocą uzyskanego modelu.

    * Otwiera w przeglądarce stronę,
    która zawiera źle sklasyfikowane
    wiersze zbioru testowego
    (`dataset/*.test.txt + dataset/*.dev.txt`)
    z podświetleniem tych tokenów,
    które wpływają dodatnio
    lub ujemnie na wynik klasyfikacji.

5. Wybrać jeden ze zbiorów danych
z folderu `dataset`, np.:

    * `hotels.sentence.*.txt` — opinie o hotelach podzielone na zdania;

    * `medicine.sentence.*.txt` — opinie o lekarzach podzielone na zdania;

    * `products.text.*.txt` — opinie o zakupach — całe wypowiedzi;

    * `reviews.text.*.txt` — opinie o nauczycielach akademickich
    — całe wypowiedzi.

6. W pliku `zadanie.py` przypisać stałej
`DATASET` jedną z wartości `'hotels.sentence'`,
`'medicine.sentence'`, `'products.text'`,
`'reviews.text'` lub inną, odpowiadającą
wybranemu zbiorowi danych.

7. Uruchomić skrypt `zadanie.py`.

8. W razie kłopotów z odróżnieniem
w przeglądarce podświetlenia wyrazów
na zielono i na czerwono odszukać w pliku
`zadanie.py` stałe `GREEN` i `RED`
i je zmienić.

9. Sprawdzić wpływ parametru `penalty`
na jakość modelu. Za miarę jakości modelu
przyjąć wartość na przecięciu wiersza
`weighted avg` i kolumny `f1-score`
(im większa liczba, tym lepiej).
Uwagi:

    * Wynikiem uczenia modelu zgodnie
    z regresją logistyczną jest słownik,
    który przypisuje współczynniki do segmentów,
    oraz *wyraz wolny* (*bias*).
    Po zastosowaniu modelu do segmentów,
    składających się na jakiś tekst,
    otrzymuje się sumę współczynników tych segmentów
    i wyrazu wolnego. Tę sumę przekształca
    się na liczbę z przedziału (0, 1)
    zgodnie z [funkcją logistyczną](https://en.wikipedia.org/wiki/Logistic_function).
    Jeśli wynik tego przekształcenia
    jest większy niż 1/2 (co zachodzi,
    gdy suma jest dodatnia), model przewiduje,
    że tekst ma wydźwięk dodatni.
    Jeśli wynik tego przekształcenia
    jest mniejszy niż 1/2 (co zachodzi,
    gdy suma jest ujemna), model przewiduje,
    że tekst ma wydźwięk ujemny.

    * Parametr `penalty` może przyjmować
    wartości `l1` lub `l2`. Odpowiada im
    regresja logistyczna z regularyzacją L1
    (zwana *lasso regression*) i regresja
    logistyczna z regularyzacją L2 (zwana
    *ridge regression*). Modele otrzymane
    z regularyzacją L1 są *rzadkie*, czyli
    nie wszystkim segmentom występującym
    w danych uczących przypisują niezerowe wagi.
    Modele otrzymane z regularyzacją L2
    są *gęste*: przypisują niezerowe wagi
    wszystkim segmentom występującym w danych
    uczących.

10. Skopiować do sprawozdania dane
obu modeli, poczynając od `{'dataset':`.
Pozostawić w programie tę wartość `penalty`,
która prowadzi do lepszych wyników.

11. Sprawdzić wpływ pomijania wyrazów
nieinformatywnych na jakość klasyfikacji.
Skopiować listę takich wyrazów
ze strony pod adresem https://pl.wikipedia.org/wiki/Wikipedia:Stopwords
do zmiennej `STOP_WORDS`.
Skopiować do sprawozdania dane modelu,
poczynając od `{'dataset':`.
Pozostawić w programie tę wartość `STOP_WORDS`,
która prowadzi do lepszych wyników.

12. Powyższe podejście jest raczej
prymitywne. Nie bierze ono pod uwagę
*polaryzacji* wyrazów, czyli tego,
że np. fragment `'nie polecam'` ma wydźwięk
przeciwny do fragmentu `'polecam'`.
Żeby temu zaradzić, proszę wykonać
następujące polecenia:

    * Dopisać do zbioru `NEGATION_START`
    wyrazy, które zmieniają wydźwięk
    swoich następników na przeciwny.
    Przykładowe wyrazy: `'nie'`, `'bez'`,
    `'oprócz'`, `'prócz'`, `'poza'`, `'brak'`,
    `'źle'`. Nie przejmujemy się tym,
    że niektóre z tych wyrazów mają też
    znaczenia niezwiązane z negacją,
    np. `'przez nie'`. Uwaga: wpisywanie na hurra
    do zbioru `NEGATION_START` wszystkiego,
    co nam przyjdzie do głowy, prowadzi donikąd.
    Zamiast tego należy na próbę dodawać do zbioru
    (początkowo pustego) kolejne pojedyncze wyrazy,
    za każdym razem sprawdzać, czy wyraz się przydał
    (poprawił jakość modelu), i odpowiednio
    zostawiać go w zbiorze albo usuwać ze zbioru.
    Takie podejście (mierzyć, mierzyć, mierzyć!)
    powinno się zresztą stosować przy wszelkich
    zagadnieniach *data science*.

    * Zmodyfikować funkcję `preprocess_tokens()`
    tak, by zamieniała wyrazy o przeciwnym wydźwięku
    na wielkie litery (segmenty pisane wielkimi
    literami umownie traktujemy jako mające wydźwięk
    przeciwny do wydźwięku ich odpowiedników pisanych
    małymi literami). Konkretnie należy iterować
    zmienną `token` po liście `tokens` i wykonywać
    poniższe kroki:

        * jeśli zmienna `negate` jest prawdziwa,
        to dodać `token.upper()` do listy `result`;

        * jeśli zmienna `negate` jest fałszywa,
        to dodać `token` do listy `result`;

        * jeśli `token` należy do zbioru `NEGATION_START`,
        to zmienić wartość zmiennej `negate` na przeciwną,
        czyli `not negate`;

        * jeśli `token` jest znakiem przestankowym,
        czyli `token.isalnum()` jest fałszywe,
        to przypisać zmiennej `negate` wartość `False`
        — dzięki tej heurystyce przeciwna polaryzacja
        wyrazów w liście `result` zamiast ciągnąć się
        bez końca, kończy się na pierwszym znaku
        przestankowym, co z grubsza odpowiada rzeczywistości.

13. Skopiować do sprawozdania treść funkcji
`preprocess_tokens()` i wartość stałej `NEGATION_START`,
dla której uzyskano najlepszy model.
Skopiować do sprawozdania dane modelu,
poczynając od `{'dataset':`.

14. Powyższy sposób traktowania polaryzacji w modelu,
w którym współczynniki wyrazów `'polecam'` i `'POLECAM'`
są od siebie zupełnie niezależne, nie jest zły,
ale z racji tego, że tylko niewielka część wyrazów
w opiniach jest zanegowana, wiele z nich nie ma
przypisanego żadnego współczynnika regresji logistycznej.
Klasa `PlusMinusVectorizer` działa podobnie do klasy
`text.CountVectorizer`, ale zawsze przypisuje wyrazom
w rodzaju `'polecam'` i `'POLECAM'` współczynniki
o przeciwnych wartościach. Proszę zastąpić przypisanie
`vectorizer = text.CountVectorizer(...)` przez
`vectorizer = PlusMinusVectorizer(...)`.
Skopiować do sprawozdania dane modelu,
poczynając od `{'dataset':`.

15. Niezaleźnie od tego, czy powyższy punkt poprawił
jakość modelu, zostawić `vectorizer = PlusMinusVectorizer(...)`,
bo tylko ta klasa jest dostosowana do zmiany wyrazów
na ich formy podstawowe. Proszę skopiować swoje funkcje
`formy_podstawowe()` i `jednoznaczna_forma_podstawowa()` z
[laboratorium 1](https://github.com/PK-PJN/laboratorium01).
Skopiować do sprawozdania dane otrzymanego modelu,
poczynając od `{'dataset':`.

16. Dodać do sprawozdania jedno zdanie podsumowania:
który z testowanych modeli najlepiej klasyfikuje
opinie z wybranego przez Państwa zbioru.

17. Zadanie nadobowiązkowe: przyjrzeć
się pięciu źle sklasyfikowanym opiniom,
wyświetlonym w przeglądarce, i jeśli
przyjdą nam do głowy jakieś wnioski
na ich temat, dopisać je do sprawozdania.

18. Zadanie nadobowiązkowe: sprawdzić
wpływ na jakość modelu parametru
`'negative_word_coefficient'`,
czyli tego, przez co trzeba pomnożyć współczynnik
np. segmentu `'polecam'`, żeby otrzymać
współczynnik segmentu `'POLECAM'`.
W najlepszym z modeli
korzystających z klasy `PlusMinusVectorizer`
pozmieniać wartość tego parametru.
Skopiować do sprawozdania dane otrzymanego modelu,
poczynając od `{'dataset':`.
Dodać do sprawozdania wniosek:
dla której wartości model był najlepszy.

19. Zadanie nadobowiązkowe: spróbować
innych metod klasyfikacji, np.
[*Support Vector Classification*](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).
Należy przy tym zakomentować wywołanie funkcji
`show_misclassifications_in_browser()`,
bo w `svm.LinearSVC` nie ma metody
`.predict_proba()`.
Skopiować do sprawozdania to,
co zostanie wypisane na ekranie,
poczynając od liczby segmentów
o niezerowych wagach.
