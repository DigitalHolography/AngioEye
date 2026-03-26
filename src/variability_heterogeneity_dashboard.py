import os
import re
import tempfile
import zipfile
from collections import defaultdict
from tkinter import Tk, filedialog
import shutil
import h5py
import warnings
import numpy as np
import pandas as pd

# --- CONFIGURATION ET CONSTANTES ---
SEGMENT_METRIC_FOLDER = "/Pipelines/arterial_waveform_shape_metrics/by_segment/"
SEGMENT_MODE = "bandlimited_segment"
EPS = 1e-12

# Utilisation de votre liste complète de métriques
SEGMENT_SELECTED_METRICS = {
    "mu_t_over_T", "RI", "PI", "R_VTI", "SF_VTI", "sigma_t_over_T", "W50_over_T",
    "E_low_over_E_total", "E_high_over_E_total", "t_max_over_T", "t_min_over_T",
    "Delta_t_over_T", "slope_rise_normalized", "slope_fall_normalized",
    "t_up_over_T", "t_down_over_T", "S_decay", "crest_factor", "R_SD",
    "Delta_DTI", "gamma_t", "spectral_entropy", "delta_phi2", "rho_h_90",
    "mu_h", "sigma_h", "N_eff_over_T", "N_H_over_T", "phase_locking_residual",
    "E_recon_H_MAX", "Q_t_skew", "Q_t_width", "R_Q_t", "Q_d_skew",
    "Q_d_width", "R_Q_d", "v_end_over_v_mean", "E_slope", "E_curv",
}

def extract_segment_metric(h5_path, metric_name, mode=SEGMENT_MODE):
    dataset_path = f"{SEGMENT_METRIC_FOLDER}{mode}/{metric_name}"
    try:
        with h5py.File(h5_path, "r") as f:
            if dataset_path not in f:
                return None
            return np.array(f[dataset_path], dtype=float)
    except Exception:
        return None

def compute_custom_stats_logic(arr):
    
    if arr is None or arr.ndim != 3:
        return None
    
    n_beats = arr.shape[0]
    beat_cvs, beat_mads, beat_iqrs = [], [], []

    for b in range(n_beats):
        
        spatial_data = arr[b, :, :].flatten()
        spatial_data = spatial_data[np.isfinite(spatial_data)]
        
        if spatial_data.size < 2:
            continue
            
        
        m = np.nanmean(spatial_data)
        s = np.nanstd(spatial_data, ddof=1)
        cv = s / (np.abs(m) + EPS)
        
        med = np.nanmedian(spatial_data)
        mad = np.nanmedian(np.abs(spatial_data - med))
        
        q75, q25 = np.nanpercentile(spatial_data, [75, 25])
        iqr = q75 - q25
        
        beat_cvs.append(cv)
        beat_mads.append(mad)
        beat_iqrs.append(iqr)

    n_beats, n_branches, n_radius = arr.shape
    segment_cv_beat = []

    for s_idx in range(n_branches):
       
        for r_idx in range(n_radius):
            
            spatial_data = arr[:, s_idx, r_idx]
                           
            m_seg = np.nanmean(spatial_data)
            s_dev_seg = np.nanstd(spatial_data, ddof=1)
            cv_seg = s_dev_seg / (np.abs(m_seg) + EPS)
            segment_cv_beat.append(cv_seg)
        
        
    

    if not beat_cvs:
        return None
        
    return {
        "CV_beat": np.nanmedian(beat_cvs),
        "MAD_beat": np.nanmedian(beat_mads),
        "IQR_beat": np.nanmedian(beat_iqrs),
        "CV_segment" : np.nanmedian(segment_cv_beat)
    }

def analyze_zip_to_df(zip_path):
    """Parcourt le zip et calcule les stats pour chaque patient/métrique."""
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        for root, _, files in os.walk(tmpdir):
            h5_files = [f for f in files if f.endswith(".h5")]
            if not h5_files:
                continue

            group_name = os.path.basename(root)
            if root == tmpdir: group_name = "Global"

            for file in h5_files:
                h5_path = os.path.join(root, file)
                for metric in SEGMENT_SELECTED_METRICS:
                    arr = extract_segment_metric(h5_path, metric)
                    stats = compute_custom_stats_logic(arr)
                    if stats:
                        rows.append({
                            "Groupe": group_name,
                            "Patient": file,
                            "Metrique": metric,
                            "CV_beat": stats["CV_beat"],
                            "MAD_beat": stats["MAD_beat"],
                            "IQR_beat": stats["IQR_beat"],
                            "CV_segment" : stats ["CV_segment"]
                        })
    return pd.DataFrame(rows)

def save_dashboard(original_zip):
    print("Analyse des données et calcul des statistiques de groupe...")
    df = analyze_zip_to_df(original_zip)
    
    if df.empty:
        print("Aucune donnée trouvée.")
        return

    with tempfile.TemporaryDirectory() as tmp_tex_dir:
        folder_name = "tableaux_latex"
        os.makedirs(os.path.join(tmp_tex_dir, folder_name))

        for group in df["Groupe"].unique():
            df_group = df[df["Groupe"] == group].copy()
            
            
            df_summary = df_group.groupby("Metrique").agg({
                "CV_beat":  ["median", "std"],
                "MAD_beat": ["median", "std"],
                "IQR_beat": ["median", "std"],
                "CV_segment": ["median", "std"]
            })

            
            df_summary.columns = [f"{col[0]}_{col[1]}" for col in df_summary.columns]
            df_summary = df_summary.reset_index()

            
            for m in ["CV_beat", "MAD_beat", "IQR_beat", "CV_segment"]:
                df_summary[m] = df_summary.apply(
                    lambda x: f"{x[m+'_median']:.3f} \\pm {x[m+'_std']:.3f}" if pd.notnull(x[m+'_std']) else f"{x[m+'_median']:.3f}", 
                    axis=1
                )

            
            df_final = df_summary[["Metrique", "CV_beat", "MAD_beat", "IQR_beat", "CV_segment"]]

           
            df_final['Metrique'] = df_final['Metrique'].replace('_', r'\_', regex=True)
            df_final.columns=[
                    "\\textbf{Métrique}", 
                    "\\textbf{Med$_b$(CV$_{seg}$) }", 
                    "\\textbf{Med$_b$(MAD$_{seg}$)}", 
                    "\\textbf{Med$_b$(IQR$_{seg}$)}",
                    "\\textbf{Med$_{seg}$(CV$_{b}$)}"
                ]

            
            tex_table = df_final.to_latex(
                index=False,
                column_format="l p{2.5cm} p{2.5cm}  p{2.5cm} p{2.5cm}",
                escape=False,
                
            )

            
            full_tex = (
                "\\begin{table}[htbp]\n"
                "\\centering\n"
                f"\\caption{{Synthèse de variabilité : Groupe {group} (Valeurs calculées par médiane des beats)}}\n"
                "\\small\n" 
                f"{tex_table}"
                "\\end{table}"
            )

            file_name = f"tableaux_{group.replace(' ', '_')}.tex"
            file_path = os.path.join(tmp_tex_dir, folder_name, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(full_tex)

       
        print(f"Mise à jour du fichier ZIP...")
        temp_zip = original_zip + ".tmp"
        with zipfile.ZipFile(original_zip, "r") as zin:
            with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    if not item.filename.startswith(folder_name + "/"):
                        zout.writestr(item, zin.read(item.filename))
                for root, _, files in os.walk(os.path.join(tmp_tex_dir, folder_name)):
                    for f in files:
                        zout.write(os.path.join(root, f), os.path.join(folder_name, f))
        os.replace(temp_zip, original_zip)
        print(f"Succès ! Fichiers disponibles dans le dossier '{folder_name}'.")

def choose_zip():
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])
    root.destroy()
    return path

if __name__ == "__main__":
    zip_p = choose_zip()
    if zip_p:
        save_dashboard(zip_p)