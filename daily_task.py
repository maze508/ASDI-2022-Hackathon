from nasa import *
from ghcnd_stations import *

# When running this script, it is assumed that your AWS S3 has data/gn/ghcnd-stations.csv. If you don't have it,
# run ghcnd_stations.py first

if __name__ == '__main__':
    # Download nasa data
    download("data/nasa/states")
    combine_states("data/nasa/states", "data/nasa")

    # Download ghcnd data (ASDI AWS dataset)
    merge_df("2022", r"data/gn/ghcnd-stations.csv", r"data/gn/years")
    combine_years(r"data/gn/years", r"data/gn")
    group_data(r"data/gn/combined.csv")
