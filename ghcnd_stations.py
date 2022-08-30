import os
import requests
import pandas as pd
from io import BytesIO
from utils.connect_to_s3 import s3


def ghcnd_station_txt_to_csv(filename, output_path):
    """
    Convert ghcnd stations data from .txt to .csv format
    :param filename: .txt file
    :param output_path: location to save output csv
    """
    id_list = []
    lat_list = []
    long_list = []
    elev_list = []
    state_list = []

    response = requests.get(filename)
    data = response.text
    for i, line in enumerate(data.split('\n')):
        if line.startswith("US"):
            data = line.split()
            id_list.append(data[0])
            lat_list.append(data[1])
            long_list.append(data[2])
            elev_list.append(data[3])
            state_list.append(data[4])

    df = pd.DataFrame()
    df["id"] = id_list
    df["lat"] = lat_list
    df["long"] = long_list
    df["elev"] = elev_list
    df["state"] = state_list
    print(df)

    output_file = output_path + "/ghcnd-stations.csv"
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer)
    s3.Object('adsi-aws-bucket', output_file).put(Body=csv_buffer.getvalue(), Key=output_file)


def merge_df(year, station_path, output_path):
    """
    Combines ghcnd stations with temperature data
    :param year: year of temperature data to download
    :param station_path: path of ghcnd csv file
    :param output_path: location to save output
    """
    print("looking at: ", year)
    print("saving to: ", os.path.join(output_path, str(year) + ".csv"))

    yearly_df = pd.read_csv(f"http://noaa-ghcn-pds.s3.amazonaws.com/csv.gz/{year}.csv.gz",
                            compression='gzip',
                            header=None,
                            names=["id", "date", "element", "value", "m-flag", "q-flag", "s-flag", "obs-time"])

    yearly_df = yearly_df[yearly_df["id"].str.startswith("US")]
    yearly_df = yearly_df[yearly_df["element"].isin(["PRCP", "TMIN", "TMAX", "TAVG"])]
    yearly_df = yearly_df[["id", "date", "element", "value"]]

    stations_df = pd.read_csv(s3.Bucket('adsi-aws-bucket').Object(station_path).get()['Body'], index_col=0)
    print(stations_df)

    yearly_df.drop_duplicates(inplace=True)

    yearly_df = yearly_df.set_index(["id", "date", "element"]).unstack()["value"]
    yearly_df.reset_index(inplace=True)

    yearly_df = yearly_df.merge(stations_df, on="id")

    print(yearly_df)

    # output_file = os.path.join(output_path, str(year) + ".csv")
    output_file = output_path + "/" + str(year) + ".csv"
    csv_buffer = BytesIO()
    yearly_df.to_csv(csv_buffer)
    s3.Object('adsi-aws-bucket', output_file).put(Body=csv_buffer.getvalue(), Key=output_file)


def combine_years(file_path, output_path):
    """
    Combine yearly temperature data into 1 csv
    :param output_path: location to save output
    :param file_path: path of all temperature csv files
    """
    combined_df = pd.DataFrame()
    for obj in s3.Bucket('adsi-aws-bucket').objects.filter(Prefix=file_path):
        df = pd.read_csv(obj.get()["Body"])
        combined_df = combined_df.append(df)
    print(combined_df)

    output_file = output_path + "/combined.csv"
    csv_buffer = BytesIO()
    combined_df.to_csv(csv_buffer)
    s3.Object('adsi-aws-bucket', output_file).put(Body=csv_buffer.getvalue(), Key=output_file)


def group_data(filename):
    """
    Group temperature csv by date and state
    :param filename: temperature csv
    """
    df = pd.read_csv(s3.Bucket('adsi-aws-bucket').Object(filename).get()['Body'], index_col=0)
    x = df.groupby(["date", "state"]).agg({"TAVG": "mean", "TMAX": "mean", "TMIN": "mean", "PRCP": "mean"})
    x.reset_index(inplace=True)

    output_file = "data/gn/aggregate.csv"
    csv_buffer = BytesIO()
    x.to_csv(csv_buffer)
    s3.Object('adsi-aws-bucket', output_file).put(Body=csv_buffer.getvalue(), Key=output_file)


if __name__ == '__main__':
    ghcnd_station_txt_to_csv("http://noaa-ghcn-pds.s3.amazonaws.com/ghcnd-stations.txt", "data/gn")
    merge_df("2022", r"data/gn/ghcnd-stations.csv", r"data/gn/years")
    combine_years(r"data/gn/years", r"data/gn")
    group_data(r"data/gn/combined.csv")
