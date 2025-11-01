# EmbedSLR - Multi-Criteria Decision Analysis (MCDA)

## 📋 Spis treści

1. [Wprowadzenie](#wprowadzenie)
2. [Nowe funkcje](#nowe-funkcje)
3. [Metody MCDA](#metody-mcda)
4. [Instalacja](#instalacja)
5. [Użycie](#użycie)
6. [Przykłady](#przykłady)
7. [API Reference](#api-reference)

## 🎯 Wprowadzenie

Rozszerzenie biblioteki EmbedSLR o funkcjonalność **analizy wielokryterialnej** (MCDA - Multi-Criteria Decision Analysis). 

Biblioteka umożliwia teraz ranking publikacji naukowych nie tylko na podstawie podobieństwa semantycznego, ale także z uwzględnieniem:
- 📚 **Słów kluczowych** - częstość występowania najważniejszych słów kluczowych
- 📖 **Referencji** - częstość cytowanych źródeł  
- 🏆 **Cytowań** - liczba cytowań artykułu
- 🧮 **Analiza wielokryterialna** - kombinacja wszystkich kryteriów z zadanymi wagami

## ✨ Nowe funkcje

### Moduł `ranking.py`

Funkcje do rankowania publikacji:

- `rank_by_keywords()` - ranking na podstawie słów kluczowych
- `rank_by_references()` - ranking na podstawie referencji
- `rank_by_citations()` - ranking na podstawie cytowań
- `compute_keyword_frequency()` - oblicza częstości słów kluczowych
- `compute_reference_frequency()` - oblicza częstości referencji
- `detailed_frequency_report()` - generuje szczegółowy raport częstości

### Moduł `mcda.py`

Metody analizy wielokryterialnej:

- `l_scoring()` - Linear Scoring (rankingowa punktowa ważona)
- `z_scoring()` - Z-Score normalization (standaryzacja)
- `l_scoring_plus()` - L-Scoring z bonusami za wartości odstające
- `mcda_report()` - generuje raport z wyników MCDA

## 📊 Metody MCDA

### 1. **L-Scoring (Linear Scoring)**

Metoda rankingowa punktowa ważona:

1. Każde kryterium jest konwertowane na ranking punktowy (najlepszy = P punktów, najgorszy = 1 punkt)
2. Punkty są mnożone przez wagi
3. Suma ważona daje końcowy wynik

**Zalety:**
- Prosta i intuicyjna
- Odporna na wartości odstające
- Łatwa interpretacja wyników

**Kiedy używać:**
- Gdy preferujesz stabilne rankingi
- Gdy wszystkie kryteria mają być traktowane równomiernie

**Przykład:**

```python
from embedslr import l_scoring

criteria = {
    "semantic": "distance_cosine",
    "keywords": "keywords_points",
    "references": "references_points",
    "citations": "citations_points"
}

weights = {
    "semantic": 0.4,
    "keywords": 0.3,
    "references": 0.2,
    "citations": 0.1
}

ascending = {
    "semantic": True,  # mniejsza odległość = lepiej
    "keywords": False,
    "references": False,
    "citations": False
}

result = l_scoring(df, criteria, weights, ascending)
```

### 2. **Z-Scoring**

Metoda oparta na standaryzacji z-score:

1. Każda wartość jest normalizowana: z = (x - μ) / σ
2. Z-scores są mnożone przez wagi
3. Suma ważona daje końcowy wynik

**Zalety:**
- Uwzględnia rozkład wartości
- Lepiej odróżnia skrajności
- Matematycznie elegancka

**Kiedy używać:**
- Gdy chcesz podkreślić różnice między artykułami
- Gdy rozkład wartości ma znaczenie

**Przykład:**

```python
from embedslr import z_scoring

result = z_scoring(df, criteria, weights, ascending)
```

### 3. **L-Scoring+ (z bonusami)**

Rozszerzenie L-Scoring o bonusy za wartości odstające:

1. Standardowy L-Scoring
2. **Bonus** jeśli artykuł jest lepszy od mediany o więcej niż `bonus_threshold` σ
3. Bonus rośnie liniowo do `max_bonus_threshold` σ
4. Maksymalny bonus = P punktów (liczba artykułów)

**Zalety:**
- Wyróżnia artykuły wybitne
- Łączy stabilność L-Scoring z nagrodą za wyjątkowość
- Można dostosować progi bonusów

**Kiedy używać:**
- Gdy chcesz promować wybitne artykuły
- Gdy szukasz "hidden gems"

**Parametry:**
- `bonus_threshold` (domyślnie 2.0) - próg dla rozpoczęcia bonusów
- `max_bonus_threshold` (domyślnie 4.0) - próg dla maksymalnego bonusu

**Przykład:**

```python
from embedslr import l_scoring_plus

result = l_scoring_plus(
    df, criteria, weights, ascending,
    bonus_threshold=2.0,
    max_bonus_threshold=4.0
)
```

## 🚀 Instalacja

```bash
# Zainstaluj zaktualizowaną wersję
pip install -e .

# Lub skopiuj nowe moduły do istniejącej instalacji
cp embedslr/mcda.py /path/to/embedslr/
cp embedslr/ranking.py /path/to/embedslr/
```

## 💻 Użycie

### Użycie przez Google Colab

```python
from embedslr import colab_run

colab_run()
```

Po uruchomieniu system zapyta:
1. Czy użyć MCDA? (y/N)
2. Wybór metody (1: L-Scoring, 2: Z-Scoring, 3: L-Scoring+)
3. Czy użyć własnych wag? (y/N)

### Użycie lokalne (wizard)

```python
from embedslr.wizard import run

run()
```

Analogicznie - interaktywny wizard poprowadzi przez wszystkie opcje.

### Użycie programistyczne

```python
import pandas as pd
from embedslr import (
    get_embeddings,
    rank_by_cosine,
    rank_by_keywords,
    rank_by_references,
    rank_by_citations,
    l_scoring_plus,
    mcda_report
)

# 1. Wczytaj dane
df = pd.read_csv("publications.csv")

# 2. Ranking semantyczny
df["combined_text"] = df["Title"] + " " + df["Abstract"]
vecs = get_embeddings(df["combined_text"].tolist(), provider="sbert")
qvec = get_embeddings(["your research query"], provider="sbert")[0]
df = rank_by_cosine(qvec, vecs, df)

# 3. Dodatkowe rankingi
df = rank_by_keywords(df, top_k=5)
df = rank_by_references(df, top_b=15)
df = rank_by_citations(df)

# 4. Analiza wielokryterialna
criteria = {
    "semantic": "distance_cosine",
    "keywords": "keywords_points",
    "references": "references_points",
    "citations": "citations_points"
}

weights = {
    "semantic": 0.4,
    "keywords": 0.3,
    "references": 0.2,
    "citations": 0.1
}

ascending = {
    "semantic": True,
    "keywords": False,
    "references": False,
    "citations": False
}

result = l_scoring_plus(df, criteria, weights, ascending)

# 5. Generuj raport
mcda_report(result, method="l_scoring_plus", path="mcda_report.txt")

# 6. Zapisz wyniki
result.to_csv("final_ranking.csv", index=False)
```

## 📖 Przykłady

Zobacz plik `examples_mcda.py` dla szczegółowych przykładów:

- Przykład 1: Podstawowe użycie
- Przykład 2: Rankingi słów kluczowych i referencji
- Przykład 3: L-Scoring
- Przykład 4: Z-Scoring
- Przykład 5: L-Scoring+
- Przykład 6: Pełny pipeline
- Przykład 7: Własne wagi

## 📚 API Reference

### `rank_by_keywords(df, top_k=5, penalty_no_keywords=0.0, fill_method="mean")`

Rankuje artykuły na podstawie słów kluczowych.

**Parametry:**
- `df` - DataFrame z kolumną 'Author Keywords'
- `top_k` - liczba najczęstszych słów do uwzględnienia
- `penalty_no_keywords` - kara za brak słów kluczowych (0.0-1.0)
- `fill_method` - metoda wypełniania ("mean", "global_mean", "zero")

**Zwraca:**
DataFrame z kolumnami: `keywords_sum`, `keywords_points`, `keywords_rank`

---

### `rank_by_references(df, top_b=15, penalty_no_refs=0.0, fill_method="mean")`

Rankuje artykuły na podstawie referencji.

**Parametry:**
- `df` - DataFrame z kolumną 'Parsed_References' lub 'References'
- `top_b` - liczba najczęstszych referencji do uwzględnienia
- `penalty_no_refs` - kara za brak referencji (0.0-1.0)
- `fill_method` - metoda wypełniania

**Zwraca:**
DataFrame z kolumnami: `references_sum`, `references_points`, `references_rank`

---

### `rank_by_citations(df)`

Rankuje artykuły na podstawie cytowań.

**Parametry:**
- `df` - DataFrame z kolumną 'Cited by' lub podobną

**Zwraca:**
DataFrame z kolumnami: `citations_points`, `citations_rank`

---

### `l_scoring(df, criteria, weights, ascending=None)`

Metoda L-Scoring.

**Parametry:**
- `df` - DataFrame z danymi
- `criteria` - dict mapujący nazwę kryterium na kolumnę
- `weights` - dict z wagami (muszą sumować się do 1.0)
- `ascending` - dict określający kierunek (True = mniejsze lepsze)

**Zwraca:**
DataFrame z kolumnami: `{criterion}_points`, `l_score`, `l_rank`

---

### `z_scoring(df, criteria, weights, ascending=None)`

Metoda Z-Scoring.

**Parametry:**
Analogiczne do `l_scoring()`

**Zwraca:**
DataFrame z kolumnami: `{criterion}_zscore`, `z_score`, `z_rank`

---

### `l_scoring_plus(df, criteria, weights, ascending=None, bonus_threshold=2.0, max_bonus_threshold=4.0)`

Metoda L-Scoring+ z bonusami.

**Parametry:**
- (jak `l_scoring()` plus:)
- `bonus_threshold` - próg σ dla rozpoczęcia bonusu
- `max_bonus_threshold` - próg σ dla maksymalnego bonusu

**Zwraca:**
DataFrame z dodatkowymi kolumnami: `{criterion}_bonus`, `total_bonus`, `l_plus_score`, `l_plus_rank`

---

### `mcda_report(df, method="l_scoring", path=None)`

Generuje raport tekstowy.

**Parametry:**
- `df` - DataFrame z wynikami MCDA
- `method` - użyta metoda ("l_scoring", "z_scoring", "l_scoring_plus")
- `path` - ścieżka do zapisu (opcjonalnie)

**Zwraca:**
String z raportem tekstowym

---

### `compute_keyword_frequency(df)`

Oblicza częstości słów kluczowych.

**Zwraca:**
Tuple (Counter, DataFrame z kolumnami: keyword, frequency)

---

### `compute_reference_frequency(df)`

Oblicza częstości referencji.

**Zwraca:**
Tuple (Counter, DataFrame z kolumnami: reference, frequency)

---

### `detailed_frequency_report(keyword_freq, reference_freq, path=None, top_n=50)`

Generuje szczegółowy raport częstości.

**Parametry:**
- `keyword_freq` - DataFrame z częstościami słów kluczowych
- `reference_freq` - DataFrame z częstościami referencji
- `path` - ścieżka do zapisu (opcjonalnie)
- `top_n` - liczba elementów do wyświetlenia

**Zwraca:**
String z raportem tekstowym

## 🎓 Dodatkowe informacje

### Publikacje naukowe

Metody MCDA zastosowane w EmbedSLR bazują na literaturze naukowej:

1. **Weighted Scoring Model**:
   - Coombes B. et al. (2015) - "Weighted Score Tests Implementing Model-Averaging Approaches"
   - Allen S. (2024) - "Weighted scoring Rules: Emphasizing Particular Outcomes"
   - Chen YT. et al. (2020) - "Development of a weighted scoring system..."

2. **Z-Score Methods**:
   - Linnen DT. et al. (2019) - "Statistical Modeling and Aggregate-Weighted Scoring..."

### Pliki wyjściowe

Po uruchomieniu MCDA otrzymasz:

**Podstawowe pliki:**
- `ranking.csv` - ranking semantyczny
- `topN.csv` - top N artykułów (jeśli wybrano)
- `biblio_report.txt` - raport bibliometryczny

**Pliki MCDA:**
- `mcda_ranking.csv` - końcowy ranking MCDA
- `mcda_topN.csv` - top N z MCDA
- `mcda_report.txt` - raport MCDA
- `keyword_frequencies.csv` - częstości słów kluczowych
- `reference_frequencies.csv` - częstości referencji
- `frequency_report.txt` - szczegółowy raport częstości

Wszystkie pliki są pakowane do `embedslr_results.zip`.

## 🤝 Wkład

EmbedSLR jest projektem open-source. Wszelkie sugestie i pull requesty są mile widziane!

## 📄 Licencja

Zgodnie z oryginalną licencją projektu EmbedSLR.

## 📧 Kontakt

W przypadku pytań dotyczących funkcjonalności MCDA, prosimy o kontakt przez Issues na GitHubie.

---

**Wersja:** 0.6.0  
**Data:** 2025-01-01  
**Autor rozszerzenia MCDA:** [Twoje dane]
