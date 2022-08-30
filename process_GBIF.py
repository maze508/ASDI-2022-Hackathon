import numpy as np
import pandas as pd
from io import BytesIO
from utils.connect_to_s3 import s3
from geopy.geocoders import Nominatim
from geopy.point import Point

# test=pd.read_csv(r"./data/rubus.csv")
obj = s3.Bucket('adsi-aws-bucket').Object('data/rubus/combined.csv').get()
test = pd.read_csv(obj['Body'], index_col=0)

test = test[(test["country"]=="United States of America") & (test["basisOfRecord"].isin(["HUMAN_OBSERVATION", "OBSERVATION", "MACHINE OBSERVATION", "OCCURRENCE"]))]
temp_test = test[test["stateProvince"].isna()]


geolocator = Nominatim(user_agent="test")

def reverse_geocoding(lat, lon):
    print(lat, lon)
    try:
        location = geolocator.reverse(Point(lat, lon))
        return location.raw['address'].get('state', '')
    except:
        return None


temp_test['address'] = temp_test.apply(lambda x: reverse_geocoding(x.decimalLatitude, x.decimalLongitude), axis=1)

new = test.merge(temp_test[["key", 'lastCrawled', 'lastParsed', 'address']], how='left', on=['key', 'lastCrawled', 'lastParsed'])
new["newstateProvince"] = new["stateProvince"].fillna(new["address"])

# new.to_csv("out_postprocess.csv", index=False)

csv_buffer = BytesIO()
new.to_csv(csv_buffer)
output_file = 'data/rubus/combined.csv'
s3.Object('adsi-aws-bucket', output_file).put(Body=csv_buffer.getvalue(), Key=output_file)
