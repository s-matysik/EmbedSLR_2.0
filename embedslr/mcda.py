"""
EmbedSLR - Multi-Criteria Decision Analysis (MCDA)
===================================================

Implementacja metod analizy wielokryterialnej:
• L-Scoring (Linear Scoring) - rankingowa punktowa ważona
• Z-Scoring - standaryzacja z użyciem odchyleń standardowych
• L-Scoring+ - L-Scoring z bonusami dla wartości odstających
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


def _validate_weights(weights: Dict[str, float]) -> None:
    """Walidacja wag - suma musi wynosić 1.0"""
    total = sum(weights.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Suma wag musi wynosić 1.0, otrzymano: {total:.6f}")


def _rank_with_ties(series: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Przypisuje punkty rangowania z uwzględnieniem remisów (średnia wartość).
    
    Parameters
    ----------
    series : pd.Series
        Wartości do rankowania
    ascending : bool
        True jeśli mniejsze wartości są lepsze (np. koszt)
        False jeśli większe wartości są lepsze (np. jakość)
    
    Returns
    -------
    pd.Series
        Punkty rankingowe (najlepszy = n punktów, najgorszy = 1 punkt)
    """
    n = len(series)
    
    # Sortowanie z zachowaniem remisów
    if ascending:
        ranks = series.rank(method='average', ascending=True)
    else:
        ranks = series.rank(method='average', ascending=False)
    
    # Przekształcenie na punkty: najlepszy rank (1) = n punktów
    points = n - ranks + 1
    
    return points


def l_scoring(
    df: pd.DataFrame,
    criteria: Dict[str, str],
    weights: Dict[str, float],
    ascending: Optional[Dict[str, bool]] = None
) -> pd.DataFrame:
    """
    Metoda L-Scoring (Linear Scoring) - rankingowa punktowa ważona.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame z danymi do analizy
    criteria : Dict[str, str]
        Słownik mapujący nazwę kryterium na nazwę kolumny w df
        np. {'semantic_similarity': 'distance_cosine', 'keywords': 'keywords_rank'}
    weights : Dict[str, float]
        Wagi dla każdego kryterium (muszą sumować się do 1.0)
        np. {'semantic_similarity': 0.4, 'keywords': 0.3, 'references': 0.3}
    ascending : Optional[Dict[str, bool]]
        Czy mniejsze wartości są lepsze dla danego kryterium
        Domyślnie: False (większe = lepsze) dla wszystkich kryteriów
        np. {'distance_cosine': True, 'keywords_rank': False}
    
    Returns
    -------
    pd.DataFrame
        Kopia df z dodatkowymi kolumnami:
        - {criterion}_points: punkty rankingowe dla każdego kryterium
        - l_score: końcowy wynik ważony
        - l_rank: pozycja w rankingu (1 = najlepszy)
    """
    _validate_weights(weights)
    
    if ascending is None:
        ascending = {k: False for k in criteria.keys()}
    
    result = df.copy()
    
    # Dla każdego kryterium oblicz punkty rankingowe
    for criterion_name, column_name in criteria.items():
        if column_name not in df.columns:
            raise ValueError(f"Kolumna '{column_name}' nie istnieje w DataFrame")
        
        is_ascending = ascending.get(criterion_name, False)
        points = _rank_with_ties(df[column_name], ascending=is_ascending)
        
        result[f"{criterion_name}_points"] = points
    
    # Oblicz wynik ważony
    weighted_sum = pd.Series(0.0, index=df.index)
    
    for criterion_name in criteria.keys():
        weight = weights[criterion_name]
        points_col = f"{criterion_name}_points"
        weighted_sum += result[points_col] * weight
    
    result['l_score'] = weighted_sum
    result['l_rank'] = result['l_score'].rank(method='min', ascending=False).astype(int)
    
    return result.sort_values('l_rank')


def z_scoring(
    df: pd.DataFrame,
    criteria: Dict[str, str],
    weights: Dict[str, float],
    ascending: Optional[Dict[str, bool]] = None
) -> pd.DataFrame:
    """
    Metoda Z-Scoring - standaryzacja przy użyciu z-score.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame z danymi do analizy
    criteria : Dict[str, str]
        Słownik mapujący nazwę kryterium na nazwę kolumny w df
    weights : Dict[str, float]
        Wagi dla każdego kryterium (muszą sumować się do 1.0)
    ascending : Optional[Dict[str, bool]]
        Czy mniejsze wartości są lepsze dla danego kryterium
        Domyślnie: False (większe = lepsze) dla wszystkich kryteriów
    
    Returns
    -------
    pd.DataFrame
        Kopia df z dodatkowymi kolumnami:
        - {criterion}_zscore: znormalizowana wartość z-score
        - z_score: końcowy wynik ważony
        - z_rank: pozycja w rankingu (1 = najlepszy)
    """
    _validate_weights(weights)
    
    if ascending is None:
        ascending = {k: False for k in criteria.keys()}
    
    result = df.copy()
    
    # Dla każdego kryterium oblicz z-score
    for criterion_name, column_name in criteria.items():
        if column_name not in df.columns:
            raise ValueError(f"Kolumna '{column_name}' nie istnieje w DataFrame")
        
        values = df[column_name]
        mean = values.mean()
        std = values.std()
        
        if std == 0:
            # Jeśli wszystkie wartości są identyczne
            z_scores = pd.Series(0.0, index=df.index)
        else:
            z_scores = (values - mean) / std
        
        # Jeśli mniejsze wartości są lepsze, odwróć znak
        if ascending.get(criterion_name, False):
            z_scores = -z_scores
        
        result[f"{criterion_name}_zscore"] = z_scores
    
    # Oblicz wynik ważony
    weighted_sum = pd.Series(0.0, index=df.index)
    
    for criterion_name in criteria.keys():
        weight = weights[criterion_name]
        zscore_col = f"{criterion_name}_zscore"
        weighted_sum += result[zscore_col] * weight
    
    result['z_score'] = weighted_sum
    result['z_rank'] = result['z_score'].rank(method='min', ascending=False).astype(int)
    
    return result.sort_values('z_rank')


def l_scoring_plus(
    df: pd.DataFrame,
    criteria: Dict[str, str],
    weights: Dict[str, float],
    ascending: Optional[Dict[str, bool]] = None,
    bonus_threshold: float = 2.0,
    max_bonus_threshold: float = 4.0
) -> pd.DataFrame:
    """
    Metoda L-Scoring+ - L-Scoring z bonusami za wartości odstające.
    
    Artykuły, które w którymkolwiek kryterium są lepsze od mediany o więcej niż
    bonus_threshold odchyleń standardowych, otrzymują dodatkowe punkty bonusowe.
    
    Bonusy są przyznawane liniowo:
    - o >= max_bonus_threshold σ: bonus = P punktów (gdzie P = liczba artykułów)
    - bonus_threshold <= o < max_bonus_threshold: bonus liniowy
    - o < bonus_threshold: brak bonusu
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame z danymi do analizy
    criteria : Dict[str, str]
        Słownik mapujący nazwę kryterium na nazwę kolumny w df
    weights : Dict[str, float]
        Wagi dla każdego kryterium (muszą sumować się do 1.0)
    ascending : Optional[Dict[str, bool]]
        Czy mniejsze wartości są lepsze dla danego kryterium
    bonus_threshold : float
        Próg odchyleń standardowych od mediany dla rozpoczęcia bonusów (domyślnie 2.0)
    max_bonus_threshold : float
        Próg odchyleń standardowych dla maksymalnego bonusu (domyślnie 4.0)
    
    Returns
    -------
    pd.DataFrame
        Kopia df z dodatkowymi kolumnami (jak w l_scoring) plus:
        - {criterion}_bonus: punkty bonusowe dla każdego kryterium
        - total_bonus: suma bonusów ze wszystkich kryteriów
        - l_plus_score: końcowy wynik (l_score + total_bonus)
        - l_plus_rank: pozycja w rankingu (1 = najlepszy)
    """
    _validate_weights(weights)
    
    if ascending is None:
        ascending = {k: False for k in criteria.keys()}
    
    # Najpierw oblicz standardowy L-Scoring
    result = l_scoring(df, criteria, weights, ascending)
    
    n = len(df)  # P = liczba artykułów
    total_bonus = pd.Series(0.0, index=df.index)
    
    # Dla każdego kryterium oblicz bonusy
    for criterion_name, column_name in criteria.items():
        values = df[column_name]
        median = values.median()
        std = values.std()
        
        if std == 0:
            # Jeśli wszystkie wartości są identyczne, brak bonusów
            result[f"{criterion_name}_bonus"] = 0.0
            continue
        
        # Oblicz odchylenie od mediany w jednostkach σ
        is_ascending = ascending.get(criterion_name, False)
        
        if is_ascending:
            # Dla ascending, lepsze wartości są mniejsze (poniżej mediany)
            deviations = (median - values) / std
        else:
            # Dla descending, lepsze wartości są większe (powyżej mediany)
            deviations = (values - median) / std
        
        # Oblicz bonus
        bonuses = pd.Series(0.0, index=df.index)
        
        for idx in df.index:
            o = deviations[idx]
            
            if o < bonus_threshold:
                bonus = 0.0
            elif o >= max_bonus_threshold:
                bonus = n  # Maksymalny bonus = P punktów
            else:
                # Liniowa interpolacja między bonus_threshold a max_bonus_threshold
                ratio = (o - bonus_threshold) / (max_bonus_threshold - bonus_threshold)
                bonus = ratio * n
            
            bonuses[idx] = bonus
        
        result[f"{criterion_name}_bonus"] = bonuses
        total_bonus += bonuses
    
    result['total_bonus'] = total_bonus
    result['l_plus_score'] = result['l_score'] + result['total_bonus']
    result['l_plus_rank'] = result['l_plus_score'].rank(method='min', ascending=False).astype(int)
    
    return result.sort_values('l_plus_rank')


def mcda_report(
    df: pd.DataFrame,
    method: str = "l_scoring",
    path: Optional[str] = None
) -> str:
    """
    Generuje raport tekstowy z wyników analizy wielokryterialnej.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame z wynikami analizy (po zastosowaniu l_scoring, z_scoring lub l_scoring_plus)
    method : str
        Użyta metoda: "l_scoring", "z_scoring" lub "l_scoring_plus"
    path : Optional[str]
        Ścieżka do zapisu raportu (opcjonalnie)
    
    Returns
    -------
    str
        Sformatowany raport tekstowy
    """
    lines = [
        "=" * 60,
        f"RAPORT ANALIZY WIELOKRYTERIALNEJ - {method.upper()}",
        "=" * 60,
        ""
    ]
    
    # Określ kolumny z wynikami na podstawie metody
    if method == "l_scoring":
        score_col = "l_score"
        rank_col = "l_rank"
    elif method == "z_scoring":
        score_col = "z_score"
        rank_col = "z_rank"
    elif method == "l_scoring_plus":
        score_col = "l_plus_score"
        rank_col = "l_plus_rank"
    else:
        raise ValueError(f"Nieznana metoda: {method}")
    
    # Statystyki podstawowe
    lines.append(f"Liczba artykułów: {len(df)}")
    lines.append(f"Średni wynik: {df[score_col].mean():.4f}")
    lines.append(f"Mediana wyniku: {df[score_col].median():.4f}")
    lines.append(f"Odchylenie standardowe: {df[score_col].std():.4f}")
    lines.append(f"Min wynik: {df[score_col].min():.4f}")
    lines.append(f"Max wynik: {df[score_col].max():.4f}")
    lines.append("")
    
    # Top 10 artykułów
    lines.append("TOP 10 ARTYKUŁÓW:")
    lines.append("-" * 60)
    
    top10 = df.head(10)
    title_col = "Title" if "Title" in df.columns else "Article Title"
    
    for idx, row in top10.iterrows():
        rank = int(row[rank_col])
        score = row[score_col]
        title = row.get(title_col, "N/A")
        
        # Ogranicz długość tytułu
        if len(str(title)) > 70:
            title = str(title)[:67] + "..."
        
        lines.append(f"{rank:3d}. [{score:7.3f}] {title}")
    
    lines.append("")
    
    # Dodatkowe informacje dla L-Scoring+
    if method == "l_scoring_plus" and "total_bonus" in df.columns:
        lines.append("BONUSY:")
        lines.append("-" * 60)
        
        bonus_df = df[df['total_bonus'] > 0].sort_values('total_bonus', ascending=False)
        
        if len(bonus_df) > 0:
            lines.append(f"Artykułów z bonusem: {len(bonus_df)}")
            lines.append(f"Średni bonus: {df['total_bonus'].mean():.4f}")
            lines.append(f"Maks. bonus: {df['total_bonus'].max():.4f}")
            lines.append("")
            lines.append("TOP 5 ARTYKUŁÓW Z NAJWYŻSZYM BONUSEM:")
            
            for idx, row in bonus_df.head(5).iterrows():
                rank = int(row[rank_col])
                bonus = row['total_bonus']
                title = row.get(title_col, "N/A")
                
                if len(str(title)) > 60:
                    title = str(title)[:57] + "..."
                
                lines.append(f"  Rank {rank:3d}, Bonus {bonus:6.2f}: {title}")
        else:
            lines.append("Żaden artykuł nie otrzymał bonusu.")
        
        lines.append("")
    
    report_text = "\n".join(lines)
    
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)
    
    return report_text


# ────────── Eksport głównych funkcji ──────────────────────
__all__ = [
    "l_scoring",
    "z_scoring", 
    "l_scoring_plus",
    "mcda_report"
]
