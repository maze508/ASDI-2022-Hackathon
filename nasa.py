import pandas as pd
from io import BytesIO
from utils.connect_to_s3 import s3
from utils.usa_converter import us_state_to_abbrev


def download(output_path):
    """
    Download nasa data by state and save to csv
    :param output_path: location to save output
    """
    df = pd.read_html("https://www.latlong.net/category/states-236-14.html")[0]
    for index, row in df.iterrows():
        state = row["Place Name"].split(",")[0]
        try:
            print(state, us_state_to_abbrev[state])
            state = us_state_to_abbrev[state]
        except:
            print("Error: ", state)
        print(state)
        lat = row["Latitude"]
        long = row["Longitude"]
        nasa = pd.read_csv(f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,ALLSKY_"
                           f"SFC_PAR_TOT,QV2M,RH2M,PS,WS2M,WS2M_MAX,WS2M_MIN,WD2M,WS10M,WS10M_MAX,WS10M_MIN,WD10M,"
                           f"GWETTOP,GWETROOT,GWETPROF&community=AG&longitude={long}&latitude={lat}&"
                           f"start=20000101&end=20220808&format=CSV", skiprows=24)
        output_file = output_path + f"/{state}.csv"
        csv_buffer = BytesIO()
        nasa.to_csv(csv_buffer)
        s3.Object('adsi-aws-bucket', output_file).put(Body=csv_buffer.getvalue(), Key=output_file)


def combine_states(file_path, output_path):
    """
    Combine states data into 1 csv
    :param file_path: path of all temperature csv files
    :param output_path: location to save output
    """
    total = pd.DataFrame()

    for obj in s3.Bucket('adsi-aws-bucket').objects.filter(Prefix=file_path):
        df = pd.read_csv(obj.get()["Body"])
        df["STATE"] = obj.key.split("/")[-1][:2]
        total = total.append(df)

    output_file = output_path + f"/nasa.csv"
    csv_buffer = BytesIO()
    total.to_csv(csv_buffer)
    s3.Object('adsi-aws-bucket', output_file).put(Body=csv_buffer.getvalue(), Key=output_file)


if __name__ == '__main__':
    download("data/nasa/states")
    combine_states("data/nasa/states", "data/nasa")
