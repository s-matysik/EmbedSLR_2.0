from __future__ import annotations
import io, os, sys, tempfile, zipfile, shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
from IPython.display import HTML, clear_output, display

IN_COLAB = "google.colab" in sys.modules


# helpers
def _env_var(p: str) -> str | None:
    return {"openai": "OPENAI_API_KEY", "cohere": "COHERE_API_KEY",
            "jina": "JINA_API_KEY", "nomic": "NOMIC_API_KEY"}.get(p.lower())


def _models() -> Dict[str, List[str]]:
    from .embeddings import list_models
    return list_models()


def _ensure_aux_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tworzy Parsed_References / Author Keywords jeżeli brak."""
    if "Parsed_References" not in df.columns:
        if "References" in df.columns:
            df["Parsed_References"] = df["References"].fillna("").apply(
                lambda x: {r.strip() for r in x.split(");") if r.strip()}
            )
        else:
            df["Parsed_References"] = [set()] * len(df)

    if "Author Keywords" not in df.columns:
        df["Author Keywords"] = ""

    # spójny Title
    if "Title" not in df.columns:
        if "Article Title" in df.columns:
            df["Title"] = df["Article Title"]
        else:
            df["Title"] = [f"Paper_{i}" for i in range(len(df))]
    return df


def _pipeline(df: pd.DataFrame, query: str, provider: str, model: str,
              out: Path, top_n: int | None, 
              use_mcda: bool = False, mcda_method: str = "l_scoring",
              mcda_weights: Dict[str, float] | None = None) -> Path:
    from .io import autodetect_columns, combine_title_abstract
    from .embeddings import get_embeddings
    from .similarity import rank_by_cosine
    from .bibliometrics import full_report
    from .ranking import (rank_by_keywords, rank_by_references, rank_by_citations,
                         compute_keyword_frequency, compute_reference_frequency,
                         detailed_frequency_report)
    from .mcda import l_scoring, z_scoring, l_scoring_plus, mcda_report

    df = _ensure_aux_columns(df.copy())
    tcol, acol = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, tcol, acol)

    vecs = get_embeddings(df["combined_text"].tolist(),
                          provider=provider, model=model)
    qvec = get_embeddings([query], provider=provider, model=model)[0]
    ranked = rank_by_cosine(qvec, vecs, df)

    p_all = out / "ranking.csv"
    ranked.to_csv(p_all, index=False)

    p_top = None
    if top_n:
        p_top = out / "topN.csv"
        ranked.head(top_n).to_csv(p_top, index=False)

    rep = out / "biblio_report.txt"
    full_report(ranked, path=rep, top_n=top_n)
    
    # New MCDA functionality
    if use_mcda:
        # Compute additional rankings
        print("⏳ Computing keyword rankings...")
        ranked = rank_by_keywords(ranked, top_k=5)
        
        print("⏳ Computing reference rankings...")
        ranked = rank_by_references(ranked, top_b=15)
        
        print("⏳ Computing citation rankings...")
        ranked = rank_by_citations(ranked)
        
        # Generate frequency reports
        print("⏳ Generating frequency reports...")
        kw_counter, kw_freq = compute_keyword_frequency(ranked)
        ref_counter, ref_freq = compute_reference_frequency(ranked)
        
        # Save frequency data
        kw_freq.to_csv(out / "keyword_frequencies.csv", index=False)
        ref_freq.to_csv(out / "reference_frequencies.csv", index=False)
        
        freq_report = out / "frequency_report.txt"
        detailed_frequency_report(kw_freq, ref_freq, path=freq_report)
        
        # Prepare criteria and weights for MCDA
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
            "semantic": True,  # distance - mniejsze lepsze
            "keywords": False,
            "references": False,
            "citations": False
        }
        
        # Apply MCDA method
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
        
        # Save MCDA results
        mcda_result.to_csv(out / "mcda_ranking.csv", index=False)
        
        if top_n:
            mcda_result.head(top_n).to_csv(out / "mcda_topN.csv", index=False)
        
        # Generate MCDA report
        mcda_rep = out / "mcda_report.txt"
        mcda_report(mcda_result, method=mcda_method, path=mcda_rep)
        
        # Update ranked to use MCDA results for final output
        ranked = mcda_result

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


# interactive Colab
def _colab_ui(out_dir: Path):
    from google.colab import files  # type: ignore

    display(HTML(
        "<h3>EmbedSLR – interactive upload (with MCDA support)</h3>"
        "<ol><li><b>Browse</b> → CSV</li><li>Wait for ✅</li>"
        "<li>Answer prompts in console</li></ol>"
    ))
    up = files.upload()
    if not up:
        display(HTML("<b style='color:red'>abort – no file</b>")); return
    name, data = next(iter(up.items()))
    df = pd.read_csv(io.BytesIO(data))
    display(HTML(f"✅ Loaded <code>{name}</code> ({len(df)} rows)<br>"))

    q = input("❓ Research query: ").strip()
    provs = list(_models())
    print("Providers:", provs)
    prov = input(f"Provider [default={provs[0]}]: ").strip() or provs[0]

    print("Models for", prov)
    for m in _models()[prov]:
        print("  •", m)
    mod = input("Model [ENTER=1st]: ").strip() or _models()[prov][0]

    n_raw = input("🔢 Top‑N for metrics [ENTER=all]: ").strip()
    top_n = int(n_raw) if n_raw else None

    key = input("API key (ENTER skip): ").strip()
    if key and (ev := _env_var(prov)):
        os.environ[ev] = key
    
    # New MCDA options
    use_mcda_input = input("🎯 Use Multi-Criteria Decision Analysis? (y/N): ").strip().lower()
    use_mcda = use_mcda_input == 'y'
    
    mcda_method = "l_scoring"
    mcda_weights = None
    
    if use_mcda:
        print("\nAvailable MCDA methods:")
        print("  1. l_scoring (Linear Scoring - default)")
        print("  2. z_scoring (Z-Score normalization)")
        print("  3. l_scoring_plus (Linear Scoring with bonuses)")
        mcda_choice = input("Choose method [1-3, default=1]: ").strip()
        
        if mcda_choice == "2":
            mcda_method = "z_scoring"
        elif mcda_choice == "3":
            mcda_method = "l_scoring_plus"
        else:
            mcda_method = "l_scoring"
        
        print(f"✓ Selected: {mcda_method}\n")
        
        custom_weights = input("Use custom weights? (y/N): ").strip().lower()
        if custom_weights == 'y':
            print("Enter weights (must sum to 1.0):")
            w_sem = float(input("  Semantic similarity [0.4]: ").strip() or "0.4")
            w_kw = float(input("  Keywords [0.3]: ").strip() or "0.3")
            w_ref = float(input("  References [0.2]: ").strip() or "0.2")
            w_cit = float(input("  Citations [0.1]: ").strip() or "0.1")
            
            mcda_weights = {
                "semantic": w_sem,
                "keywords": w_kw,
                "references": w_ref,
                "citations": w_cit
            }
            print(f"✓ Custom weights set\n")

    print("⏳ Computing …")
    zip_tmp = _pipeline(df, q, prov, mod, out_dir, top_n, 
                       use_mcda, mcda_method, mcda_weights)

    dst = Path.cwd() / zip_tmp.name
    shutil.copy(zip_tmp, dst)
    print("✅ Finished – downloading ZIP")
    files.download(str(dst))


# CLI fallback
def _cli(out_dir: Path):
    print("== EmbedSLR CLI ==")
    csv_p = Path(input("CSV path: ").strip())
    df = pd.read_csv(csv_p)
    q = input("Query: ").strip()
    prov = input("Provider [sbert]: ").strip() or "sbert"
    mod = input("Model [ENTER=default]: ").strip() or _models()[prov][0]
    n_raw = input("Top‑N [ENTER=all]: ").strip()
    top_n = int(n_raw) if n_raw else None
    key = input("API key [skip]: ").strip()
    if key and (ev := _env_var(prov)):
        os.environ[ev] = key
    
    # MCDA options for CLI
    use_mcda_input = input("Use Multi-Criteria Decision Analysis? (y/N): ").strip().lower()
    use_mcda = use_mcda_input == 'y'
    
    mcda_method = "l_scoring"
    mcda_weights = None
    
    if use_mcda:
        print("\nAvailable MCDA methods:")
        print("  1. l_scoring (Linear Scoring)")
        print("  2. z_scoring (Z-Score)")
        print("  3. l_scoring_plus (L-Scoring+)")
        mcda_choice = input("Choose [1-3, default=1]: ").strip()
        
        if mcda_choice == "2":
            mcda_method = "z_scoring"
        elif mcda_choice == "3":
            mcda_method = "l_scoring_plus"
    
    z = _pipeline(df, q, prov, mod, out_dir, top_n, use_mcda, mcda_method, mcda_weights)
    print("ZIP saved:", z)


# public
def run(save_dir: str | os.PathLike | None = None):
    save_dir = Path(save_dir or tempfile.mkdtemp(prefix="embedslr_"))
    clear_output()
    (_colab_ui if IN_COLAB else _cli)(save_dir)
