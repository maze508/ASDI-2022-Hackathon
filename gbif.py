import requests
import pandas as pd
from io import BytesIO
from utils.connect_to_s3 import s3


def get_rubus_dataset(year):
    """
    Download Rebus phoenicolasius dataset from GBIF based on year
    :param year: year number
    """
    df = pd.DataFrame()

    parameters = {
        "year": year,
        "taxon_key": "2995149",
        "limit": 300
    }

    offset = 0
    while True:

        parameters['offset'] = offset
        response = requests.get("https://api.gbif.org/v1/occurrence/search", params=parameters).json()
        total = response['count']

        try:
            df = df.append(pd.DataFrame(response['results']))
            print(f"{offset} of {total}", len(df))
        except KeyError as e:
            print(f"ERROR: {offset} of {total}, {e}")

        if response['endOfRecords']:
            break
        offset += 300

    output_file = f"data/rubus/years/{str(year)}.csv"
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer)
    s3.Object('adsi-aws-bucket', output_file).put(Body=csv_buffer.getvalue(), Key=output_file)


def combine_rebus():
    """"
    Combine different years of rebus dataset into a single csv
    """
    combined_df = pd.DataFrame()
    for obj in s3.Bucket('adsi-aws-bucket').objects.filter(Prefix="data/rubus/years"):
        df = pd.read_csv(obj.get()["Body"])
        combined_df = combined_df.append(df)

    output_file = "data/rubus/combined.csv"
    csv_buffer = BytesIO()
    combined_df.to_csv(csv_buffer)
    s3.Object('adsi-aws-bucket', output_file).put(Body=csv_buffer.getvalue(), Key=output_file)
    return


if __name__ == '__main__':
    get_rubus_dataset(2022)
    combine_rebus()

