import os

import pandas as pd

from astropy import units as u
from astropy.time import Time

from dk154_targets.query_managers import TNSQueryManager

from dk154_targets import paths


tns_archive_path = paths.archive_path / "tns"
tns_archive_path.mkdir(exist_ok=True, parents=True)



start_date = Time("2019-04-01")
end_date = Time("2022-09-30")
interval = 40 * u.day

qm = TNSQueryManager.from_default_config({})
ii = 0
df_list = []
interval_df_paths = []

while True:
    search_start = start_date + ii * interval
    search_end = start_date + (ii + 1) * interval - 1 * u.day

    params = {
        "classified_sne": "0", 
        "format": "csv", 
        "date_start[date]": search_start.iso.split()[0], # Strip date from YYYY-mm-DD HH:MM:SS
        "date_end[date]": search_end.iso.split()[0],
        "redshift_min": "0.00",
        "redshift_max": "3.00",
    }
    interval_df_path = tns_archive_path / f"interval_df_{ii:03d}.csv"

    print(params)
    if not interval_df_path.exists():
        interval_df = qm.perform_query(params, sleep_time=2.5)
        interval_df.to_csv(interval_df_path, index=False)

    else:
        print("read existing df")
        try:
            interval_df = pd.read_csv(interval_df_path)
        except pd.errors.EmptyDataError:
            interval_df = None

    if interval_df is not None:
        df_list.append(interval_df)
        interval_df_paths.append(interval_df_path)


    ii = ii + 1
    if search_end > end_date:
        break

df = pd.concat(df_list)
print(df.columns)
df.sort_values("ID", inplace=True)


outpath = tns_archive_path / "unclassified_sne.csv"
df.to_csv(outpath, index=False)

for interval_df_path in interval_df_paths:
    print(interval_df_path)
    os.remove(interval_df_path)
    assert not interval_df_path.exists()