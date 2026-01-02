import argparse
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm

# Import our library functions and utilities
from plot_utils import setup_plot_style, STEADY_COLORS, save_figure

# Suppress warnings from xarray about datetime formats
warnings.filterwarnings("ignore", category=UserWarning, module="xarray")

def load_and_preprocess_argo_data(filepath: str) -> tuple:
    """
    Loads and preprocesses the Argo ocean float dataset.

    This function performs the following steps:
    1. Loads the NetCDF file and filters to North Atlantic region at 10m depth
    2. Stacks spatial dimensions and filters for points with complete time series
    3. Calculates monthly temperature anomalies (deviations from climatology)
    4. Estimates physical parameters (lambda, sigma_ou_sq) for each location
    5. Computes total observation variances
    6. Saves processed data to CSV

    Args:
        filepath (str): The path to the raw Argo NetCDF data file.

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): The final clean DataFrame with columns:
                - lon, lat: Geographic coordinates
                - Y: Mean temperature anomaly
                - lambda_est: Estimated reversion rate
                - sigma_ou_sq_est: Estimated process noise variance
                - sigma_i_sq: Total observation variance
            - temp_clean (xr.DataArray): The clean 3D data cube of temperatures
                                        (for visualization purposes).
            - anomalies (xr.DataArray): The stacked DataArray of monthly anomalies
                                       (for visualization purposes).
    """
    print("--- Starting Data Preprocessing ---")
    ds = xr.open_dataset(filepath, decode_times=True)
    print(f"Initial data loaded. Full dataset dimensions: {ds.dims}")

    # --- 1. Filter Data by Region and Depth ---
    temp_region = ds['TEMP'].sel(latitude=slice(20, 60), longitude=slice(-80, -10))
    temp_10m = temp_region.sel(depth=10, method='nearest')
    print(f"1. Data filtered for North Atlantic at 10m depth. Shape: {temp_10m.shape}")

    # --- 2. Stack and Filter for Valid Points ---
    # First, stack the spatial dimensions into a single 'point' dimension
    stacked_temp = temp_10m.stack(point=('longitude', 'latitude'))
    print(f"2. Data stacked into points. Shape: {stacked_temp.shape}")
    
    # Create a boolean mask for points that have sufficient data (are not all NaN)
    valid_points_mask = stacked_temp.notnull().sum(dim='time') == len(ds['time'])
    print(f"3. Created mask for valid points. Number of valid ocean points found: {valid_points_mask.sum().item()}")
    
    # Use the mask to select only the valid time series from the stacked data
    clean_stacked_temp = stacked_temp.sel(point=valid_points_mask)
    print(f"4. Applied mask. Shape of clean stacked data: {clean_stacked_temp.shape}")

    # --- 5. Calculate Anomalies ---
    climatology = clean_stacked_temp.groupby('time.month').mean('time')
    anomalies = clean_stacked_temp.groupby('time.month') - climatology
    Y = anomalies.mean(dim='time')
    print(f"5. Calculated mean anomalies (Y values). Total time series: {len(Y)}")

    results = []
    print("6. Estimating physical parameters for each time series...")
    # Iterate directly over the xarray 'point' coordinates for robust selection
    for point_coord in tqdm(Y.point.values, desc="Processing points"):
        series = anomalies.sel(point=point_coord).to_pandas().dropna()
        if len(series) < 12: continue

        lag1_autocorr = series.autocorr(lag=1)
        if pd.isna(lag1_autocorr) or lag1_autocorr < 0.1 or lag1_autocorr > 0.95:
            lag1_autocorr = 0.7
        lambda_est = -np.log(lag1_autocorr)
        ar1_residuals = series[1:] - lag1_autocorr * series[:-1]
        sigma_ou_sq_est = np.var(ar1_residuals)

        # Unpack the coordinate tuple for the DataFrame
        lon, lat = point_coord
        results.append({
            'lon': lon, 'lat': lat, 'Y': Y.sel(point=point_coord).item(),
            'lambda_est': lambda_est, 'sigma_ou_sq_est': sigma_ou_sq_est
        })
    
    df = pd.DataFrame(results).dropna()
    print("Parameter estimation complete.")

    # --- 7. Calculate Total Observation Variance ---
    sigma_obs_sq = 0.01**2
    df['sigma_i_sq'] = (df['sigma_ou_sq_est'] / (2 * df['lambda_est'])) + sigma_obs_sq
    print("7. Calculated total observation variances.")
    
    # --- Save the processed data for future use ---
    output_path = Path("data") / "argo_processed_data.csv"
    df.to_csv(output_path, index=False, float_format='%.10f')
    print(f"8. Processed data saved to '{output_path}'")

    print(f"\nPreprocessing complete. Final number of valid float locations: {len(df)}")
    print("--- End Data Preprocessing ---\n")
    
    return df, clean_stacked_temp, anomalies

def visualize_preprocessing(df: pd.DataFrame, temp_clean: xr.DataArray, anomalies: xr.DataArray):
    """
    Creates a three-panel diagnostic figure for the data processing pipeline.

    The figure shows:
    - Panel (a): Map of valid float locations with example points highlighted
    - Panel (b): Raw temperature time series for example points
    - Panel (c): Calculated monthly anomalies for example points

    This visualization helps verify that the preprocessing steps are working correctly.

    Args:
        df (pd.DataFrame): The processed DataFrame with float locations.
        temp_clean (xr.DataArray): The clean temperature data cube.
        anomalies (xr.DataArray): The monthly anomaly data.
    """
    print("Generating visualization figure...")
    setup_plot_style()
    fig = plt.figure(figsize=(24, 7))
    
    # --- Select 3 example points to highlight ---
    example_points = df.iloc[[
        int(len(df) * 0.25),
        int(len(df) * 0.5),
        int(len(df) * 0.75),
        int(len(df) * 0.9),
    ]]

    # --- Panel (a): Map of Valid Float Locations ---
    ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax1.scatter(df['lon'], df['lat'], color='gray', s=5, alpha=0.5, transform=ccrs.PlateCarree())
    for _, point in example_points.iterrows():
        ax1.scatter(point['lon'], point['lat'], s=50, transform=ccrs.PlateCarree(), zorder=10)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.LAND, zorder=5, edgecolor='black')
    ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax1.set_title("a) Float Locations (gray) and Example Points (color)")

    # --- Panel (b): Raw Temperature Time Series ---
    ax2 = fig.add_subplot(1, 3, 2)
    for _, point in example_points.iterrows():
        temp_clean.sel(point=(point['lon'], point['lat'])).plot(ax=ax2, label=f"Point: Lat: {point['lat']:.1f}, Lon: {point['lon']:.1f}")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Temperature (°C)")
    ax2.legend()
    ax2.grid(True, which="both", ls="--")
    ax2.set_title("b) Raw Monthly Temperatures (Example Points)")

    # --- Panel (c): Anomaly Time Series ---
    ax3 = fig.add_subplot(1, 3, 3)
    for _, point in example_points.iterrows():
        # Re-stack to select by point for anomalies
        series = anomalies.sel(longitude=point['lon'], latitude=point['lat'], method='nearest')
        series.plot(ax=ax3, label=f"Point: Lat: {point['lat']:.1f}, Lon: {point['lon']:.1f}")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Temperature Anomaly (°C)")
    ax3.legend()
    ax3.grid(True, which="both", ls="--")
    ax3.set_title("c) Calculated Monthly Anomalies (Example Points)")
    
    plt.tight_layout()
    save_figure(fig, "argo_visualization")

def visualize_physical_parameters(df: pd.DataFrame):
    """
    Creates a two-panel figure showing the spatial distribution of the
    estimated physical parameters (lambda and sigma_ou_sq).

    This visualization provides insight into the physical properties of the dataset,
    showing how stability and process noise vary spatially across the ocean region.

    Args:
        df (pd.DataFrame): The final processed DataFrame containing the
                           estimated parameters.
    """
    print("Generating physical parameter visualization figure...")
    setup_plot_style()
    fig = plt.figure(figsize=(16, 7))

    # --- Panel (a): Map of Estimated Stability (lambda) ---
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1.set_title(r'a) Estimated Physical Stability ($\lambda_{est}$)')
    
    # Use a perceptually uniform colormap
    sc1 = ax1.scatter(df['lon'], df['lat'], c=df['lambda_est'], cmap='viridis', 
                      s=15, transform=ccrs.PlateCarree(), vmin=0, vmax=np.percentile(df['lambda_est'], 95))
    
    fig.colorbar(sc1, ax=ax1, orientation='horizontal', label='Stability (Higher = More Stable)', pad=0.1)

    # --- Panel (b): Map of Estimated Process Noise (sigma_ou_sq) ---
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax2.set_title(r'b) Estimated Process Noise ($\sigma^2_{ou, est}$)')
    
    sc2 = ax2.scatter(df['lon'], df['lat'], c=df['sigma_ou_sq_est'], cmap='plasma', 
                      s=15, transform=ccrs.PlateCarree(), vmin=0, vmax=np.percentile(df['sigma_ou_sq_est'], 95))

    fig.colorbar(sc2, ax=ax2, orientation='horizontal', label='Noise Variance (Higher = Noisier Process)', pad=0.1)

    # --- Add common map features ---
    for ax in [ax1, ax2]:
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, zorder=100, edgecolor='black')
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    plt.tight_layout()
    save_figure(fig, "argo_physical_parameters_map")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and visualize Argo ocean float data.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the Argo NetCDF data file."
    )
    args = parser.parse_args()
    
    df_processed, temp_data, anomaly_data = load_and_preprocess_argo_data(args.data_path)
    
    if not df_processed.empty:
        visualize_preprocessing(df_processed, temp_data, anomaly_data.unstack())
        visualize_physical_parameters(df_processed)
