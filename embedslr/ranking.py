"""
EmbedSLR - Ranking Module
==========================

Moduł do obliczania rankingów na podstawie:
• Słów kluczowych (keywords)
• Referencji (references)
• Cytowań (citations)
"""

from __future__ import annotations

from typing import Dict, List, Set, Optional, Tuple
from collections import Counter
import pandas as pd
import numpy as np


def _parse_keywords(series: pd.Series) -> List[Set[str]]:
    """Konwertuje kolumnę 'Author Keywords' na listę setów (małe litery)."""
    return [
        {w.strip().lower() for w in str(x).split(";") if w.strip()}
        for x in series.fillna("")
    ]


def _parse_references(series: pd.Series) -> List[Set[str]]:
    """Konwertuje kolumnę 'Parsed_References' lub 'References' na listę setów."""
    result = []
    for x in series:
        if isinstance(x, set):
            result.append(x)
        elif isinstance(x, str):
            result.append({r.strip() for r in x.split(");") if r.strip()})
        else:
            result.append(set())
    return result


def compute_keyword_frequency(df: pd.DataFrame) -> Tuple[Counter, pd.DataFrame]:
    """
    Oblicza częstość występowania każdego słowa kluczowego.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame z kolumną 'Author Keywords'
    
    Returns
    -------
    Counter
        Zliczenia dla każdego słowa kluczowego
    pd.DataFrame
        DataFrame z kolumnami: keyword, frequency, sortowany malejąco
    """
    if "Author Keywords" not in df.columns:
        return Counter(), pd.DataFrame(columns=["keyword", "frequency"])
    
    keywords_sets = _parse_keywords(df["Author Keywords"])
    counter = Counter()
    
    for kw_set in keywords_sets:
        counter.update(kw_set)
    
    # Konwersja na DataFrame
    freq_df = pd.DataFrame([
        {"keyword": kw, "frequency": count}
        for kw, count in counter.most_common()
    ])
    
    return counter, freq_df


def compute_reference_frequency(df: pd.DataFrame) -> Tuple[Counter, pd.DataFrame]:
    """
    Oblicza częstość występowania każdej referencji.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame z kolumną 'Parsed_References' lub 'References'
    
    Returns
    -------
    Counter
        Zliczenia dla każdej referencji
    pd.DataFrame
        DataFrame z kolumnami: reference, frequency, sortowany malejąco
    """
    ref_col = None
    if "Parsed_References" in df.columns:
        ref_col = "Parsed_References"
    elif "References" in df.columns:
        ref_col = "References"
    
    if ref_col is None:
        return Counter(), pd.DataFrame(columns=["reference", "frequency"])
    
    references_sets = _parse_references(df[ref_col])
    counter = Counter()
    
    for ref_set in references_sets:
        counter.update(ref_set)
    
    # Konwersja na DataFrame
    freq_df = pd.DataFrame([
        {"reference": ref, "frequency": count}
        for ref, count in counter.most_common()
    ])
    
    return counter, freq_df


def rank_by_keywords(
    df: pd.DataFrame,
    top_k: int = 5,
    penalty_no_keywords: float = 0.0,
    fill_method: str = "mean"
) -> pd.DataFrame:
    """
    Rankuje artykuły na podstawie częstości występowania słów kluczowych.
    
    Algorytm:
    1. Oblicz częstość występowania każdego słowa kluczowego w całym zbiorze
    2. Dla każdego artykułu:
       - Weź top_k najczęstszych słów kluczowych z tego artykułu
       - Zsumuj ich częstości
       - Jeśli artykuł ma mniej niż top_k słów, uzupełnij średnią
    3. Przypisz punkty rankingowe (P punktów dla najlepszego, P-1 dla drugiego, itd.)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame z kolumną 'Author Keywords'
    top_k : int
        Liczba najczęstszych słów kluczowych do uwzględnienia (domyślnie 5)
    penalty_no_keywords : float
        Procent kary za brak słów kluczowych (0.0 - 1.0). Domyślnie 0.0 (bez kary)
    fill_method : str
        Metoda wypełniania brakujących slotów:
        - "mean": średnia z istniejących słów kluczowych artykułu
        - "global_mean": średnia z całego zbioru
        - "zero": wypełnij zerami
    
    Returns
    -------
    pd.DataFrame
        Kopia df z dodatkowymi kolumnami:
        - keywords_sum: suma częstości top_k słów kluczowych
        - keywords_points: punkty rankingowe
        - keywords_rank: pozycja w rankingu (1 = najlepszy)
    """
    result = df.copy()
    
    # Oblicz częstości słów kluczowych
    kw_counter, _ = compute_keyword_frequency(df)
    
    if not kw_counter:
        # Brak słów kluczowych w zbiorze
        result["keywords_sum"] = 0.0
        result["keywords_points"] = 0.0
        result["keywords_rank"] = 1
        return result
    
    keywords_sets = _parse_keywords(df["Author Keywords"])
    n = len(df)
    
    # Oblicz sumy dla każdego artykułu
    sums = []
    
    for kw_set in keywords_sets:
        if not kw_set:
            # Brak słów kluczowych w artykule
            if fill_method == "global_mean":
                avg = np.mean(list(kw_counter.values()))
                article_sum = avg * top_k
            else:
                article_sum = 0.0
        else:
            # Pobierz częstości dla słów kluczowych artykułu
            freqs = [kw_counter.get(kw, 0) for kw in kw_set]
            freqs.sort(reverse=True)
            
            # Weź top_k
            top_freqs = freqs[:top_k]
            
            # Uzupełnij brakujące sloty
            if len(top_freqs) < top_k:
                missing = top_k - len(top_freqs)
                
                if fill_method == "mean" and top_freqs:
                    fill_value = np.mean(top_freqs)
                elif fill_method == "global_mean":
                    fill_value = np.mean(list(kw_counter.values()))
                else:  # "zero"
                    fill_value = 0.0
                
                top_freqs.extend([fill_value] * missing)
            
            article_sum = sum(top_freqs)
        
        # Zastosuj karę za brak słów kluczowych
        if not kw_set and penalty_no_keywords > 0:
            article_sum *= (1.0 - penalty_no_keywords)
        
        sums.append(article_sum)
    
    result["keywords_sum"] = sums
    
    # Oblicz punkty rankingowe z uwzględnieniem remisów
    sum_series = pd.Series(sums, index=df.index)
    ranks = sum_series.rank(method='average', ascending=False)
    points = n - ranks + 1
    
    result["keywords_points"] = points
    result["keywords_rank"] = sum_series.rank(method='min', ascending=False).astype(int)
    
    return result.sort_values("keywords_rank")


def rank_by_references(
    df: pd.DataFrame,
    top_b: int = 15,
    penalty_no_refs: float = 0.0,
    fill_method: str = "mean"
) -> pd.DataFrame:
    """
    Rankuje artykuły na podstawie częstości występowania referencji.
    
    Algorytm identyczny jak dla słów kluczowych, ale dla referencji.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame z kolumną 'Parsed_References' lub 'References'
    top_b : int
        Liczba najczęstszych referencji do uwzględnienia (domyślnie 15)
    penalty_no_refs : float
        Procent kary za brak referencji (0.0 - 1.0). Domyślnie 0.0 (bez kary)
    fill_method : str
        Metoda wypełniania brakujących slotów (jak w rank_by_keywords)
    
    Returns
    -------
    pd.DataFrame
        Kopia df z dodatkowymi kolumnami:
        - references_sum: suma częstości top_b referencji
        - references_points: punkty rankingowe
        - references_rank: pozycja w rankingu (1 = najlepszy)
    """
    result = df.copy()
    
    # Oblicz częstości referencji
    ref_counter, _ = compute_reference_frequency(df)
    
    if not ref_counter:
        # Brak referencji w zbiorze
        result["references_sum"] = 0.0
        result["references_points"] = 0.0
        result["references_rank"] = 1
        return result
    
    ref_col = "Parsed_References" if "Parsed_References" in df.columns else "References"
    references_sets = _parse_references(df[ref_col])
    n = len(df)
    
    # Oblicz sumy dla każdego artykułu
    sums = []
    
    for ref_set in references_sets:
        if not ref_set:
            # Brak referencji w artykule
            if fill_method == "global_mean":
                avg = np.mean(list(ref_counter.values()))
                article_sum = avg * top_b
            else:
                article_sum = 0.0
        else:
            # Pobierz częstości dla referencji artykułu
            freqs = [ref_counter.get(ref, 0) for ref in ref_set]
            freqs.sort(reverse=True)
            
            # Weź top_b
            top_freqs = freqs[:top_b]
            
            # Uzupełnij brakujące sloty
            if len(top_freqs) < top_b:
                missing = top_b - len(top_freqs)
                
                if fill_method == "mean" and top_freqs:
                    fill_value = np.mean(top_freqs)
                elif fill_method == "global_mean":
                    fill_value = np.mean(list(ref_counter.values()))
                else:  # "zero"
                    fill_value = 0.0
                
                top_freqs.extend([fill_value] * missing)
            
            article_sum = sum(top_freqs)
        
        # Zastosuj karę za brak referencji
        if not ref_set and penalty_no_refs > 0:
            article_sum *= (1.0 - penalty_no_refs)
        
        sums.append(article_sum)
    
    result["references_sum"] = sums
    
    # Oblicz punkty rankingowe z uwzględnieniem remisów
    sum_series = pd.Series(sums, index=df.index)
    ranks = sum_series.rank(method='average', ascending=False)
    points = n - ranks + 1
    
    result["references_points"] = points
    result["references_rank"] = sum_series.rank(method='min', ascending=False).astype(int)
    
    return result.sort_values("references_rank")


def rank_by_citations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rankuje artykuły na podstawie liczby cytowań.
    
    Zakłada, że DataFrame ma kolumnę 'Cited by' lub podobną z liczbą cytowań.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame z kolumną zawierającą liczbę cytowań
    
    Returns
    -------
    pd.DataFrame
        Kopia df z dodatkowymi kolumnami:
        - citations_points: punkty rankingowe
        - citations_rank: pozycja w rankingu (1 = najlepszy)
    """
    result = df.copy()
    
    # Znajdź kolumnę z cytowaniami
    citation_col = None
    possible_cols = ["Cited by", "Citations", "Times Cited", "Citation Count"]
    
    for col in possible_cols:
        if col in df.columns:
            citation_col = col
            break
    
    if citation_col is None:
        # Brak kolumny z cytowaniami
        result["citations_points"] = 0.0
        result["citations_rank"] = 1
        return result
    
    n = len(df)
    citations = df[citation_col].fillna(0).astype(float)
    
    # Oblicz punkty rankingowe z uwzględnieniem remisów
    ranks = citations.rank(method='average', ascending=False)
    points = n - ranks + 1
    
    result["citations_points"] = points
    result["citations_rank"] = citations.rank(method='min', ascending=False).astype(int)
    
    return result.sort_values("citations_rank")


def detailed_frequency_report(
    keyword_freq: pd.DataFrame,
    reference_freq: pd.DataFrame,
    path: Optional[str] = None,
    top_n: int = 50
) -> str:
    """
    Generuje szczegółowy raport z częstościami słów kluczowych i referencji.
    
    Parameters
    ----------
    keyword_freq : pd.DataFrame
        DataFrame z częstościami słów kluczowych (z compute_keyword_frequency)
    reference_freq : pd.DataFrame
        DataFrame z częstościami referencji (z compute_reference_frequency)
    path : Optional[str]
        Ścieżka do zapisu raportu (opcjonalnie)
    top_n : int
        Liczba najczęstszych elementów do wyświetlenia (domyślnie 50)
    
    Returns
    -------
    str
        Sformatowany raport tekstowy
    """
    lines = [
        "=" * 80,
        "SZCZEGÓŁOWY RAPORT CZĘSTOŚCI WYSTĘPOWANIA",
        "=" * 80,
        ""
    ]
    
    # Słowa kluczowe
    lines.append("NAJCZĘSTSZE SŁOWA KLUCZOWE:")
    lines.append("-" * 80)
    
    if len(keyword_freq) > 0:
        lines.append(f"Całkowita liczba unikalnych słów kluczowych: {len(keyword_freq)}")
        lines.append(f"Całkowita liczba wystąpień: {keyword_freq['frequency'].sum()}")
        lines.append("")
        lines.append(f"TOP {min(top_n, len(keyword_freq))} SŁÓW KLUCZOWYCH:")
        lines.append("")
        
        for idx, row in keyword_freq.head(top_n).iterrows():
            lines.append(f"  {row['frequency']:4d}x | {row['keyword']}")
    else:
        lines.append("Brak słów kluczowych w zbiorze.")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    
    # Referencje
    lines.append("NAJCZĘSTSZE REFERENCJE:")
    lines.append("-" * 80)
    
    if len(reference_freq) > 0:
        lines.append(f"Całkowita liczba unikalnych referencji: {len(reference_freq)}")
        lines.append(f"Całkowita liczba wystąpień: {reference_freq['frequency'].sum()}")
        lines.append("")
        lines.append(f"TOP {min(top_n, len(reference_freq))} REFERENCJI:")
        lines.append("")
        
        for idx, row in reference_freq.head(top_n).iterrows():
            ref = row['reference']
            # Ogranicz długość referencji dla czytelności
            if len(ref) > 100:
                ref = ref[:97] + "..."
            lines.append(f"  {row['frequency']:4d}x | {ref}")
    else:
        lines.append("Brak referencji w zbiorze.")
    
    lines.append("")
    
    report_text = "\n".join(lines)
    
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)
    
    return report_text


# ────────── Eksport głównych funkcji ──────────────────────
__all__ = [
    "compute_keyword_frequency",
    "compute_reference_frequency",
    "rank_by_keywords",
    "rank_by_references",
    "rank_by_citations",
    "detailed_frequency_report"
]
