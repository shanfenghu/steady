# Data for the STEADY Paper Experiments

This directory is intended to hold the data files required to run the real-world case study in `experiments/exp_argo.py`.

## Argo Ocean Data

The experiment uses the monthly gridded Argo product from the Copernicus Marine Service.

1.  **Data Store:** Navigate to the Copernicus Marine Data Store: <https://data.marine.copernicus.eu/>

2.  **Search:** Use the search bar and enter the following **Product Identifier**:
    `INSITU_GLO_PHY_TS_OA_MY_013_052`

3.  **Dataset:** This will lead you to the "Global Ocean- Delayed Mode gridded CORA- In-situ Observations objective analysis in Delayed Mode" product.

4.  **Download Settings:** When downloading the data, you must specify the following settings to obtain the correct subset and a manageable file size:

    * **Date Range:** Select the time period from `1980-01-01` to the latest available date (e.g., `2024-06-01`).

    * **Geographical Area:** Specify the North Atlantic bounding box:

        * North: `60`

        * West: `-80`

        * South: `20`

        * East: `-10`

    * **Depth Range:** Select the near-surface levels (e.g., `1m` to `10m`).

5.  **Required File:** Download the resulting NetCDF (`.nc`) file.

6.  **Setup:** Place the downloaded `.nc` file into this `data/` directory.

### How to Run the Experiment

Once the data file is in place, you can run the experiment from the root of the project directory with the following command, replacing `your_argo_file.nc` with the actual filename:
```bash
python experiments/exp_argo.py --data_path data/your_argo_file.nc