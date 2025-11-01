"""
EmbedSLR – Terminal Wizard (local)
==================================
Interactive wizard for running EmbedSLR in a local environment.
The pipeline (embedding → ranking → full bibliometric report → ZIP).
"""

from __future__ import annotations

import os
import sys
import zipfile
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ────────── helper functions  ─────────────────
def _env_var(provider: str) -> str | None:
    """Returns the ENV variable name for the API key of the given provider."""
    return {
        "openai": "OPENAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "jina":   "JINA_API_KEY",
        "nomic":  "NOMIC_API_KEY",
    }.get(provider.lower())


def _ensure_sbert_installed() -> None:
    """
    Ensures the *sentence‑transformers* library is available.
    • If missing, prompts the user and installs it (`pip install --user sentence-transformers`).
    """
    try:
        importlib.import_module("sentence_transformers")
    except ModuleNotFoundError:
        ans = _ask(
            "📦  Brak biblioteki 'sentence‑transformers'. Zainstalować teraz? (y/N)",
            "N",
        ).lower()
        if ans == "y":
            print("⏳  Instaluję 'sentence‑transformers'…")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--user", "--quiet", "sentence-transformers"]
            )
            print("✅  Instalacja zakończona.\n")
        else:
            sys.exit("❌  Provider 'sbert' wymaga biblioteki 'sentence‑transformers'.")


def _models() -> Dict[str, List[str]]:
    from .embeddings import list_models
    return list_models()


def _ensure_aux_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures presence of columns:
      • Title
      • Author Keywords
      • Parsed_References  (set[str])
    """
    if "Parsed_References" not in df.columns:
        if "References" in df.columns:
            df["Parsed_References"] = df["References"].fillna("").apply(
                lambda x: {r.strip() for r in x.split(");") if r.strip()}
            )
        else:
            df["Parsed_References"] = [set()] * len(df)

    if "Author Keywords" not in df.columns:
        df["Author Keywords"] = ""

    if "Title" not in df.columns:
        if "Article Title" in df.columns:
            df["Title"] = df["Article Title"]
        else:
            df["Title"] = [f"Paper_{i}" for i in range(len(df))]
    return df


# ────────── local‑model utilities for SBERT ──────────────────────────────
def _local_model_dir(model_name: str) -> Path:
    """
    Returns a path where the given SBERT model should live inside the project
    (…/embedslr/sbert_models/<model_name_with__>).
    """
    safe = model_name.replace("/", "__")
    base = Path(__file__).resolve().parent / "sbert_models"
    return base / safe


def _get_or_download_local_sbert(model_name: str) -> Path:
    """
    Ensures that *model_name* is present in the project folder and returns its path.
    If missing – downloads it once and saves permanently.
    """
    local_dir = _local_model_dir(model_name)
    if local_dir.exists():
        print(f"✅  Lokalny model znaleziony: {local_dir}")
    else:
        print(f"⏳  Pobieram model '{model_name}' do '{local_dir}' …")
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(model_name).save(str(local_dir))
        print("✅  Model pobrany i zapisany.\n")
    # wymuszenie trybu offline dla HuggingFace Hub
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return local_dir


# ────────── core pipeline ────────────────────────────────────────────────
def _pipeline(
    df: pd.DataFrame,
    query: str,
    provider: str,
    model: str,
    out: Path,
    top_n: int | None,
    use_mcda: bool = False,
    mcda_method: str = "l_scoring",
    mcda_weights: Optional[Dict[str, float]] = None,
) -> Path:
    """
    Executes the full EmbedSLR workflow and returns the path to the ZIP of results.
    """
    from .io import autodetect_columns, combine_title_abstract
    from .embeddings import get_embeddings
    from .similarity import rank_by_cosine
    from .bibliometrics import full_report
    from .ranking import (rank_by_keywords, rank_by_references, rank_by_citations,
                         compute_keyword_frequency, compute_reference_frequency,
                         detailed_frequency_report)
    from .mcda import l_scoring, z_scoring, l_scoring_plus, mcda_report

    df = _ensure_aux_columns(df.copy())

    # 1. Prepare text for embedding
    tcol, acol = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, tcol, acol)

    # 2. Embeddings
    vecs = get_embeddings(df["combined_text"].tolist(),
                          provider=provider, model=model)
    qvec = get_embeddings([query], provider=provider, model=model)[0]

    # 3. Ranking
    ranked = rank_by_cosine(qvec, vecs, df)

    # 4. Save ranking.csv
    out.mkdir(parents=True, exist_ok=True)
    p_all = out / "ranking.csv"
    ranked.to_csv(p_all, index=False)

    # 5. Top‑N (optional)
    p_top = None
    if top_n:
        p_top = out / "topN.csv"
        ranked.head(top_n).to_csv(p_top, index=False)

    # 6. Full bibliometric report
    rep = out / "biblio_report.txt"
    full_report(ranked, path=rep, top_n=top_n)
    
    # New MCDA functionality
    if use_mcda:
        print("⏳ Computing keyword rankings...")
        ranked = rank_by_keywords(ranked, top_k=5)
        
        print("⏳ Computing reference rankings...")
        ranked = rank_by_references(ranked, top_b=15)
        
        print("⏳ Computing citation rankings...")
        ranked = rank_by_citations(ranked)
        
        print("⏳ Generating frequency reports...")
        kw_counter, kw_freq = compute_keyword_frequency(ranked)
        ref_counter, ref_freq = compute_reference_frequency(ranked)
        
        kw_freq.to_csv(out / "keyword_frequencies.csv", index=False)
        ref_freq.to_csv(out / "reference_frequencies.csv", index=False)
        
        freq_report = out / "frequency_report.txt"
        detailed_frequency_report(kw_freq, ref_freq, path=freq_report)
        
        if mcda_weights is None:
            mcda_weights = {
                "semantic": 0.4,
                "keywords": 0.3,
                "references": 0.2,
                "citations": 0.1
            }
        
        criteria = {
            "semantic": "distance_cosine",
            "keywords": "keywords_points",
            "references": "references_points",
            "citations": "citations_points"
        }
        
        ascending = {
            "semantic": True,
            "keywords": False,
            "references": False,
            "citations": False
        }
        
        print(f"⏳ Applying {mcda_method}...")
        if mcda_method == "l_scoring":
            mcda_result = l_scoring(ranked, criteria, mcda_weights, ascending)
        elif mcda_method == "z_scoring":
            mcda_result = z_scoring(ranked, criteria, mcda_weights, ascending)
        elif mcda_method == "l_scoring_plus":
            mcda_result = l_scoring_plus(ranked, criteria, mcda_weights, ascending,
                                        bonus_threshold=2.0, max_bonus_threshold=4.0)
        else:
            raise ValueError(f"Unknown MCDA method: {mcda_method}")
        
        mcda_result.to_csv(out / "mcda_ranking.csv", index=False)
        
        if top_n:
            mcda_result.head(top_n).to_csv(out / "mcda_topN.csv", index=False)
        
        mcda_rep = out / "mcda_report.txt"
        mcda_report(mcda_result, method=mcda_method, path=mcda_rep)
        
        ranked = mcda_result

    # 7. ZIP with results
    zf = out / "embedslr_results.zip"
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(p_all, "ranking.csv")
        if p_top:
            z.write(p_top, "topN.csv")
        z.write(rep, "biblio_report.txt")
        
        if use_mcda:
            z.write(out / "mcda_ranking.csv", "mcda_ranking.csv")
            if top_n:
                z.write(out / "mcda_topN.csv", "mcda_topN.csv")
            z.write(out / "mcda_report.txt", "mcda_report.txt")
            z.write(out / "keyword_frequencies.csv", "keyword_frequencies.csv")
            z.write(out / "reference_frequencies.csv", "reference_frequencies.csv")
            z.write(out / "frequency_report.txt", "frequency_report.txt")
    
    return zf


# ────────── simple CLI ────────────────────────────────────────────────────
def _ask(prompt: str, default: Optional[str] = None) -> str:
    msg = f"{prompt}"
    if default is not None:
        msg += f" [{default}]"
    msg += ": "
    ans = input(msg).strip()
    return ans or (default or "")


def _select_provider() -> str:
    provs = list(_models())
    print("📜  Available providers:", ", ".join(provs))
    return _ask("Provider", provs[0])


def _select_model(provider: str) -> str:
    mods = _models()[provider]
    print(f"📜  Models for {provider} (first 20):")
    for m in mods[:20]:
        print("   •", m)
    return _ask("Model", mods[0])


def run(save_dir: str | os.PathLike | None = None):
    """
    Runs the EmbedSLR wizard in terminal/screen/tmux.
    """
    print("\n== EmbedSLR Wizard (local) ==\n")

    # Input file
    csv_path = Path(_ask("📄  Path to CSV file")).expanduser()
    if not csv_path.exists():
        sys.exit(f"❌  File not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"✅  Loaded {len(df)} records\n")

    # Analysis parameters
    query = _ask("❓  Research query").strip()
    provider = _select_provider()

    # SBERT prerequisites
    if provider.lower() == "sbert":
        _ensure_sbert_installed()

    # Model (prompt only ONCE)
    model_name = _select_model(provider)

    # For SBERT – ensure permanent local copy & switch to its path
    if provider.lower() == "sbert":
        model_path = _get_or_download_local_sbert(model_name)
        model = str(model_path)          # use local path in the pipeline
    else:
        model = model_name               # non‑SBERT providers unchanged

    n_raw = _ask("🔢  Top‑N publications for metrics (ENTER = all)")
    top_n = int(n_raw) if n_raw else None

    # API key (if needed)
    key_env = _env_var(provider)
    if key_env and not os.getenv(key_env):
        key = _ask(f"🔑  {key_env} (ENTER = skip)")
        if key:
            os.environ[key_env] = key
    
    # New MCDA options
    use_mcda_input = _ask("🎯  Use Multi-Criteria Decision Analysis? (y/N)", "N").lower()
    use_mcda = use_mcda_input == 'y'
    
    mcda_method = "l_scoring"
    mcda_weights = None
    
    if use_mcda:
        print("\n📊  Available MCDA methods:")
        print("  1. l_scoring (Linear Scoring - default)")
        print("  2. z_scoring (Z-Score normalization)")
        print("  3. l_scoring_plus (Linear Scoring with bonuses)")
        
        mcda_choice = _ask("Choose method", "1")
        
        if mcda_choice == "2":
            mcda_method = "z_scoring"
        elif mcda_choice == "3":
            mcda_method = "l_scoring_plus"
        else:
            mcda_method = "l_scoring"
        
        print(f"✓  Selected: {mcda_method}\n")
        
        custom_weights_input = _ask("Use custom weights? (y/N)", "N").lower()
        if custom_weights_input == 'y':
            print("📏  Enter weights (must sum to 1.0):")
            w_sem = float(_ask("    Semantic similarity", "0.4"))
            w_kw = float(_ask("    Keywords", "0.3"))
            w_ref = float(_ask("    References", "0.2"))
            w_cit = float(_ask("    Citations", "0.1"))
            
            mcda_weights = {
                "semantic": w_sem,
                "keywords": w_kw,
                "references": w_ref,
                "citations": w_cit
            }
            total = sum(mcda_weights.values())
            print(f"✓  Weights set (sum = {total:.4f})\n")

    # Output folder
    out_dir = Path(save_dir or os.getcwd()).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    print("\n⏳  Processing…")
    zip_path = _pipeline(
        df=df,
        query=query,
        provider=provider,
        model=model,
        out=out_dir,
        top_n=top_n,
        use_mcda=use_mcda,
        mcda_method=mcda_method,
        mcda_weights=mcda_weights,
    )

    print("\n✅  Done!")
    print("📁  Results saved to:", out_dir)
    print("🎁  ZIP package:", zip_path)
    if use_mcda:
        print("   (ranking.csv, topN.csv, biblio_report.txt,")
        print("    mcda_ranking.csv, mcda_report.txt, frequency reports)\n")
    else:
        print("   (ranking.csv, topN.csv – if selected, biblio_report.txt)\n")


if __name__ == "__main__":
    run()
