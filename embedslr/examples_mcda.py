"""
EmbedSLR MCDA Usage Example
============================

Przykład użycia nowej funkcjonalności analizy wielokryterialnej.
"""

import pandas as pd
from embedslr import (
    get_embeddings,
    rank_by_cosine,
    rank_by_keywords,
    rank_by_references,
    rank_by_citations,
    l_scoring,
    z_scoring,
    l_scoring_plus,
    mcda_report,
    compute_keyword_frequency,
    compute_reference_frequency,
    detailed_frequency_report
)

# ═══════════════════════════════════════════════════════════════════
# Przykład 1: Podstawowe użycie - tylko semantic ranking
# ═══════════════════════════════════════════════════════════════════

def example_basic():
    """Przykład podstawowego użycia bez MCDA."""
    print("=" * 60)
    print("PRZYKŁAD 1: Podstawowe użycie (bez MCDA)")
    print("=" * 60)
    
    # Wczytaj dane
    df = pd.read_csv("publications.csv")
    
    # Przygotuj tekst do embeddingu
    df["combined_text"] = df["Title"].fillna("") + " " + df["Abstract"].fillna("")
    
    # Uzyskaj embeddingi
    query = "machine learning applications in healthcare"
    vecs = get_embeddings(df["combined_text"].tolist(), provider="sbert")
    qvec = get_embeddings([query], provider="sbert")[0]
    
    # Ranking semantyczny
    ranked = rank_by_cosine(qvec, vecs, df)
    
    print(f"\nTop 5 artykułów (semantic ranking):")
    print(ranked[["Title", "distance_cosine"]].head())
    
    return ranked


# ═══════════════════════════════════════════════════════════════════
# Przykład 2: Ranking po słowach kluczowych i referencjach
# ═══════════════════════════════════════════════════════════════════

def example_rankings():
    """Przykład użycia rankingów słów kluczowych i referencji."""
    print("\n" + "=" * 60)
    print("PRZYKŁAD 2: Rankingi słów kluczowych i referencji")
    print("=" * 60)
    
    df = pd.read_csv("publications.csv")
    
    # Oblicz częstości
    kw_counter, kw_freq = compute_keyword_frequency(df)
    ref_counter, ref_freq = compute_reference_frequency(df)
    
    print(f"\nTop 10 najczęstszych słów kluczowych:")
    print(kw_freq.head(10))
    
    # Zapisz raport częstości
    detailed_frequency_report(kw_freq, ref_freq, path="frequency_report.txt")
    
    # Rankuj artykuły
    df = rank_by_keywords(df, top_k=5)
    df = rank_by_references(df, top_b=15)
    df = rank_by_citations(df)
    
    print(f"\nTop 5 artykułów (keyword ranking):")
    print(df[["Title", "keywords_rank", "keywords_sum"]].head())
    
    return df


# ═══════════════════════════════════════════════════════════════════
# Przykład 3: L-Scoring (Linear Scoring)
# ═══════════════════════════════════════════════════════════════════

def example_l_scoring():
    """Przykład użycia metody L-Scoring."""
    print("\n" + "=" * 60)
    print("PRZYKŁAD 3: L-Scoring (Linear Scoring)")
    print("=" * 60)
    
    # Przygotuj dane z wszystkimi rankingami
    df = pd.read_csv("publications.csv")
    
    # Załóżmy, że mamy już obliczone wszystkie rankingi
    # (w rzeczywistości trzeba je obliczyć jak w przykładzie 1 i 2)
    
    # Zdefiniuj kryteria
    criteria = {
        "semantic": "distance_cosine",
        "keywords": "keywords_points",
        "references": "references_points",
        "citations": "citations_points"
    }
    
    # Zdefiniuj wagi (muszą sumować się do 1.0)
    weights = {
        "semantic": 0.4,
        "keywords": 0.3,
        "references": 0.2,
        "citations": 0.1
    }
    
    # Określ kierunek (mniejsze lepsze / większe lepsze)
    ascending = {
        "semantic": True,   # distance - mniejsze lepsze
        "keywords": False,  # punkty - większe lepsze
        "references": False,
        "citations": False
    }
    
    # Zastosuj L-Scoring
    result = l_scoring(df, criteria, weights, ascending)
    
    print(f"\nTop 10 artykułów (L-Scoring):")
    print(result[["Title", "l_score", "l_rank"]].head(10))
    
    # Wygeneruj raport
    report = mcda_report(result, method="l_scoring", path="l_scoring_report.txt")
    print("\nRaport zapisany do: l_scoring_report.txt")
    
    return result


# ═══════════════════════════════════════════════════════════════════
# Przykład 4: Z-Scoring
# ═══════════════════════════════════════════════════════════════════

def example_z_scoring():
    """Przykład użycia metody Z-Scoring."""
    print("\n" + "=" * 60)
    print("PRZYKŁAD 4: Z-Scoring")
    print("=" * 60)
    
    df = pd.read_csv("publications.csv")
    
    # Te same kryteria i wagi jak w L-Scoring
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
    
    # Zastosuj Z-Scoring
    result = z_scoring(df, criteria, weights, ascending)
    
    print(f"\nTop 10 artykułów (Z-Scoring):")
    print(result[["Title", "z_score", "z_rank"]].head(10))
    
    # Wygeneruj raport
    report = mcda_report(result, method="z_scoring", path="z_scoring_report.txt")
    print("\nRaport zapisany do: z_scoring_report.txt")
    
    return result


# ═══════════════════════════════════════════════════════════════════
# Przykład 5: L-Scoring+ (z bonusami)
# ═══════════════════════════════════════════════════════════════════

def example_l_scoring_plus():
    """Przykład użycia metody L-Scoring+ z bonusami."""
    print("\n" + "=" * 60)
    print("PRZYKŁAD 5: L-Scoring+ (z bonusami)")
    print("=" * 60)
    
    df = pd.read_csv("publications.csv")
    
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
    
    # Zastosuj L-Scoring+ z bonusami
    # Bonus zaczyna się przy 2 odchyleniach standardowych
    # Maksymalny bonus przy 4 odchyleniach standardowych
    result = l_scoring_plus(
        df, criteria, weights, ascending,
        bonus_threshold=2.0,
        max_bonus_threshold=4.0
    )
    
    print(f"\nTop 10 artykułów (L-Scoring+):")
    print(result[["Title", "l_plus_score", "l_plus_rank", "total_bonus"]].head(10))
    
    # Artykuły z najwyższymi bonusami
    bonus_articles = result[result["total_bonus"] > 0].sort_values("total_bonus", ascending=False)
    print(f"\nArtykuły z bonusami (top 5):")
    print(bonus_articles[["Title", "total_bonus", "l_plus_rank"]].head(5))
    
    # Wygeneruj raport
    report = mcda_report(result, method="l_scoring_plus", path="l_scoring_plus_report.txt")
    print("\nRaport zapisany do: l_scoring_plus_report.txt")
    
    return result


# ═══════════════════════════════════════════════════════════════════
# Przykład 6: Pełny pipeline (wszystko razem)
# ═══════════════════════════════════════════════════════════════════

def example_full_pipeline():
    """Przykład pełnego pipeline z wszystkimi krokami."""
    print("\n" + "=" * 60)
    print("PRZYKŁAD 6: Pełny pipeline MCDA")
    print("=" * 60)
    
    # 1. Wczytaj dane
    df = pd.read_csv("publications.csv")
    print(f"Wczytano {len(df)} artykułów")
    
    # 2. Ranking semantyczny
    print("\n1. Obliczam ranking semantyczny...")
    df["combined_text"] = df["Title"].fillna("") + " " + df["Abstract"].fillna("")
    query = "deep learning in medical image analysis"
    
    vecs = get_embeddings(df["combined_text"].tolist(), provider="sbert")
    qvec = get_embeddings([query], provider="sbert")[0]
    df = rank_by_cosine(qvec, vecs, df)
    
    # 3. Rankingi dodatkowe
    print("2. Obliczam ranking słów kluczowych...")
    df = rank_by_keywords(df, top_k=5)
    
    print("3. Obliczam ranking referencji...")
    df = rank_by_references(df, top_b=15)
    
    print("4. Obliczam ranking cytowań...")
    df = rank_by_citations(df)
    
    # 4. Analiza wielokryterialna
    print("5. Wykonuję analizę wielokryterialną (L-Scoring+)...")
    
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
    
    # 5. Zapisz wyniki
    print("6. Zapisuję wyniki...")
    result.to_csv("mcda_final_ranking.csv", index=False)
    result.head(20).to_csv("mcda_top20.csv", index=False)
    
    # 6. Generuj raporty
    print("7. Generuję raporty...")
    mcda_report(result, method="l_scoring_plus", path="mcda_final_report.txt")
    
    kw_counter, kw_freq = compute_keyword_frequency(result)
    ref_counter, ref_freq = compute_reference_frequency(result)
    detailed_frequency_report(kw_freq, ref_freq, path="frequency_final_report.txt")
    
    print("\n✅ Pipeline zakończony!")
    print("\nWygenerowane pliki:")
    print("  • mcda_final_ranking.csv")
    print("  • mcda_top20.csv")
    print("  • mcda_final_report.txt")
    print("  • frequency_final_report.txt")
    
    return result


# ═══════════════════════════════════════════════════════════════════
# Przykład 7: Własne wagi
# ═══════════════════════════════════════════════════════════════════

def example_custom_weights():
    """Przykład z niestandardowymi wagami."""
    print("\n" + "=" * 60)
    print("PRZYKŁAD 7: Własne wagi kryteriów")
    print("=" * 60)
    
    df = pd.read_csv("publications.csv")
    
    # Scenariusz 1: Priorytet dla semantyki
    print("\nScenariusz 1: Priorytet dla podobieństwa semantycznego")
    weights_semantic = {
        "semantic": 0.7,
        "keywords": 0.15,
        "references": 0.1,
        "citations": 0.05
    }
    
    # Scenariusz 2: Priorytet dla cytowań
    print("Scenariusz 2: Priorytet dla liczby cytowań")
    weights_citations = {
        "semantic": 0.2,
        "keywords": 0.2,
        "references": 0.2,
        "citations": 0.4
    }
    
    # Scenariusz 3: Równe wagi
    print("Scenariusz 3: Równe wagi dla wszystkich kryteriów")
    weights_equal = {
        "semantic": 0.25,
        "keywords": 0.25,
        "references": 0.25,
        "citations": 0.25
    }
    
    # Porównaj wyniki...
    # (implementacja analogiczna do poprzednich przykładów)
    
    print("\nMożesz eksperymentować z różnymi wagami w zależności od")
    print("priorytetów Twojego badania!")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║   EmbedSLR - Multi-Criteria Decision Analysis (MCDA)  ║
    ║                 Usage Examples                         ║
    ╚════════════════════════════════════════════════════════╝
    
    Uruchom przykłady definiując funkcje:
    
    • example_basic()          - podstawowe użycie
    • example_rankings()       - rankingi słów kluczowych i referencji
    • example_l_scoring()      - metoda L-Scoring
    • example_z_scoring()      - metoda Z-Scoring  
    • example_l_scoring_plus() - metoda L-Scoring+ z bonusami
    • example_full_pipeline()  - pełny pipeline
    • example_custom_weights() - własne wagi
    
    Aby uruchomić, odkomentuj poniższe linie:
    """)
    
    # Odkomentuj aby uruchomić przykłady:
    # example_basic()
    # example_rankings()
    # example_l_scoring()
    # example_z_scoring()
    # example_l_scoring_plus()
    # example_full_pipeline()
    # example_custom_weights()
