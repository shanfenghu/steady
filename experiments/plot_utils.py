import matplotlib.pyplot as plt
from pathlib import Path

# 1. Central Color Palette for consistency across all figures
STEADY_COLORS = {
    'truth': '#00A087',      # Teal
    'mle': '#3C5488',        # Dark Blue
    'gjs': '#F39B7F',         # Peach
    'steady': '#8491B4',     # Slate
    'error': '#DC0000',       # Red
    'path_js': '#B09C85',      # Taupe
    'path_steady': '#91D1C2', # Light Teal
    'ou_paths': plt.cm.cividis
}

def setup_plot_style():
    """
    Sets the global matplotlib rcParams for a consistent, professional look.
    
    Configures font sizes, styles, and other plotting parameters to ensure
    all figures have a uniform appearance. This should be called before
    creating any plots.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        # "font.family": "serif",
        # "font.serif": ["Charter", "Bitstream Charter", "STIXGeneral"],
        "text.usetex": False,
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "mathtext.fontset": "stix",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

def save_figure(fig, filename: str, output_dir: str = "figures"):
    """
    Saves a matplotlib figure in multiple high-quality formats.

    Args:
        fig: The matplotlib figure object to save.
        filename (str): The base name for the file (e.g., "risk_vs_n").
        output_dir (str, optional): The directory to save the figures in.
                                    Defaults to "figures".
    """
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Define file paths
    pdf_path = Path(output_dir) / f"{filename}.pdf"
    png_path = Path(output_dir) / f"{filename}.png"

    # Save in PDF
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')

    print(f"Figure saved to '{pdf_path}'")