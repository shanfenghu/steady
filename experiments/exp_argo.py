import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree
from tqdm import tqdm
from pathlib import Path

# Import our library functions and utilities
from steady.estimators import mle_estimator, gjs_estimator, steady_estimator
from plot_utils import setup_plot_style, STEADY_COLORS, save_figure

# Suppress warnings from cartopy about future changes
warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_proxy_metrics(df: pd.DataFrame, estimate_col: str) -> dict:
    """
    Calculates proxy metrics for evaluating the quality of estimates.

    Computes two metrics:
    1. Spatial variability: Measures how much estimates vary relative to their
       spatial neighbors (lower is better, indicating smoother estimates).
    2. Correlation with climatology: Measures correlation with latitude and longitude,
       which serve as proxies for expected climatological patterns.

    Args:
        df (pd.DataFrame): DataFrame containing float locations (lon, lat) and estimates.
        estimate_col (str): The name of the column with the estimates to evaluate.

    Returns:
        dict: A dictionary containing:
            - 'variability' (float): Spatial variability metric.
            - 'correlation_lat' (float): Correlation with latitude.
            - 'correlation_lon' (float): Correlation with longitude.
    """
    # --- 1. Spatial Variability ---
    coords = df[['lon', 'lat']].values
    estimates = df[estimate_col].values
    
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=6)
    
    neighbor_indices = indices[:, 1:]
    neighbor_estimates = estimates[neighbor_indices]
    mean_neighbor_estimate = np.mean(neighbor_estimates, axis=1)
    
    spatial_variability = np.mean((estimates - mean_neighbor_estimate)**2) / np.mean((mean_neighbor_estimate)**2)

    # --- 2. Correlation with Climatology (Proxy) ---
    correlation_lat = df[estimate_col].corr(df['lat'])
    correlation_lon = df[estimate_col].corr(df['lon'])
    
    return {'variability': spatial_variability, 
            'correlation_lat': correlation_lat, 'correlation_lon': correlation_lon}

def plot_argo_maps(df: pd.DataFrame, output_filename: str):
    """
    Generates and saves a side-by-side map comparison of the three estimators.
    
    Creates a three-panel figure showing the spatial distribution of estimates
    from MLE, GJS, and STEADY on a world map projection. All three panels use
    the same color scale for fair comparison.
    
    Args:
        df (pd.DataFrame): DataFrame containing float locations and estimates
                           (est_mle, est_gjs, est_steady columns).
        output_filename (str): Base filename for saving the figure (without extension).
    """
    setup_plot_style()
    fig = plt.figure(figsize=(24, 7))
    
    # Define a common, symmetric color scale for both plots
    clim = np.percentile(np.abs(df['est_mle']), 95)

    ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax1.set_title("a) MLE Estimates")
    sc1 = ax1.scatter(df['lon'], df['lat'], c=df['est_mle'], cmap='coolwarm', s=15, 
                      transform=ccrs.PlateCarree(), vmin=-clim, vmax=clim)
    
    ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
    ax2.set_title("b) GJS Estimates")
    sc2 = ax2.scatter(df['lon'], df['lat'], c=df['est_gjs'], cmap='coolwarm', s=15, 
                      transform=ccrs.PlateCarree(), vmin=-clim, vmax=clim)
    
    ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
    ax3.set_title("c) STEADY Estimates")
    sc3 = ax3.scatter(df['lon'], df['lat'], c=df['est_steady'], cmap='coolwarm', s=15, 
                      transform=ccrs.PlateCarree(), vmin=-clim, vmax=clim)
    
    for ax in [ax1, ax2, ax3]:
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, zorder=100, edgecolor='black')
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        
    fig.colorbar(sc2, ax=[ax1, ax2, ax3], orientation='horizontal', label='Equilibrium Temperature Anomaly (°C)', pad=0.1, extend='both')
    
    save_figure(fig, output_filename)

def main(processed_data_path: str, n_bootstraps: int = 100):
    """
    Main function to orchestrate the Argo ocean float case study.

    This function:
    1. Loads pre-processed Argo data
    2. Computes estimates using MLE, GJS, and STEADY
    3. Generates spatial visualization maps
    4. Performs bootstrap resampling to compute quantitative metrics
    5. Saves results to CSV files

    Args:
        processed_data_path (str): Path to the processed Argo CSV data file.
        n_bootstraps (int): Number of bootstrap resamples for computing metrics.
                           Defaults to 100.
    """
    print(f"Loading processed data from '{processed_data_path}'...")
    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{processed_data_path}'.")
        print("Please run 'experiments/process_argo.py' first to generate this file.")
        return

    if df.empty:
        print("Processed data file is empty. Exiting.")
        return

    print(f"Successfully loaded {len(df)} data points.")

    # --- Run Estimators on the Full Dataset ---
    df['est_mle'] = mle_estimator(df['Y'].values)
    df['est_gjs'] = gjs_estimator(df['Y'].values, df['sigma_i_sq'].values)
    
    mu_phys = df['Y'].mean()
    # mu_phys = df['Y'].median()
    df['est_steady'] = steady_estimator(
        df['Y'].values, df['lambda_est'].values, mu_phys,
        df['sigma_ou_sq_est'].mean(),
        # df['sigma_ou_sq_est'].median(),
        0.01**2
    )
    output_path = Path("results") / "argo_estimates.csv"
    df.to_csv(output_path, index=False, float_format='%.10f')
    
    # --- Generate the Main Figure ---
    print("Generating main results map...")
    plot_argo_maps(df, "argo_estimate_maps")
    
    # --- Run Bootstrapping for Quantitative Metrics ---
    print(f"\nRunning bootstrapping with {n_bootstraps} resamples...")
    bootstrap_results = []
    for i in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        df_sample = df.sample(n=len(df), replace=True, random_state=i)
        
        sample_mle = mle_estimator(df_sample['Y'].values)
        sample_gjs = gjs_estimator(df_sample['Y'].values, df_sample['sigma_i_sq'].values)
        sample_mu_phys = df_sample['Y'].mean()
        # sample_mu_phys = df_sample['Y'].median()
        sample_steady = steady_estimator(
            df_sample['Y'].values, df_sample['lambda_est'].values, sample_mu_phys,
            df_sample['sigma_ou_sq_est'].mean(), 0.01**2
            # df_sample['sigma_ou_sq_est'].median(), 0.01**2
        )
        
        df_sample['est_mle'] = sample_mle
        df_sample['est_gjs'] = sample_gjs
        df_sample['est_steady'] = sample_steady
        
        metrics_mle = calculate_proxy_metrics(df_sample, 'est_mle')
        metrics_gjs = calculate_proxy_metrics(df_sample, 'est_gjs')
        metrics_steady = calculate_proxy_metrics(df_sample, 'est_steady')
        
        bootstrap_results.append({
            'mle_variability': metrics_mle['variability'], 'mle_corr_lat': metrics_mle['correlation_lat'], 'mle_corr_lon': metrics_mle['correlation_lon'],
            'gjs_variability': metrics_gjs['variability'], 'gjs_corr_lat': metrics_gjs['correlation_lat'], 'gjs_corr_lon': metrics_gjs['correlation_lon'],
            'steady_variability': metrics_steady['variability'], 'steady_corr_lat': metrics_steady['correlation_lat'], 'steady_corr_lon': metrics_steady['correlation_lon'],
        })
        
    results_df = pd.DataFrame(bootstrap_results)
    
    summary = {
        'Estimator': ['MLE', 'GJS', 'STEADY'],
        'Spatial Variability': [
            f"{results_df['mle_variability'].mean():.3f} ± {results_df['mle_variability'].std():.3f}",
            f"{results_df['gjs_variability'].mean():.3f} ± {results_df['gjs_variability'].std():.3f}",
            f"{results_df['steady_variability'].mean():.3f} ± {results_df['steady_variability'].std():.3f}",
        ],
        'Correlation w/ Climatology (Lat)': [
            f"{results_df['mle_corr_lat'].mean():.3f} ± {results_df['mle_corr_lat'].std():.3f}",
            f"{results_df['gjs_corr_lat'].mean():.3f} ± {results_df['gjs_corr_lat'].std():.3f}",
            f"{results_df['steady_corr_lat'].mean():.3f} ± {results_df['steady_corr_lat'].std():.3f}",
        ],
        'Correlation w/ Climatology (Lon)': [
            f"{results_df['mle_corr_lon'].mean():.3f} ± {results_df['mle_corr_lon'].std():.3f}",
            f"{results_df['gjs_corr_lon'].mean():.3f} ± {results_df['gjs_corr_lon'].std():.3f}",
            f"{results_df['steady_corr_lon'].mean():.3f} ± {results_df['steady_corr_lon'].std():.3f}",
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    print("\n--- Quantitative Results ---")
    print(summary_df)
    
    Path("results").mkdir(exist_ok=True)
    summary_df.to_csv("results/argo_quantitative_results.csv", index=False)
    print("\nResults table saved to results/argo_quantitative_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Argo real-world case study.")
    parser.add_argument(
        "--processed_data_path",
        type=str,
        default="data/argo_processed_data.csv",
        help="Path to the processed Argo CSV data file."
    )
    args = parser.parse_args()
    main(processed_data_path=args.processed_data_path)
