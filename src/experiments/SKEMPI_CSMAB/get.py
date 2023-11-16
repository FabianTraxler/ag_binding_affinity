import argparse
import requests
import sys
import pandas as pd
import time


URL_single = "https://biosig.lab.uq.edu.au/csm_ab/api/prediction_single"
URL_pose = "https://biosig.lab.uq.edu.au/csm_ab/api/prediction_pose"


def main(args):
    job_type = args.job_type
    job_id = args.job_id

    params = {
        "job_id": job_id,
    }

    if job_type == 'single':
        req = requests.get(URL_single, data=params)
        # print(req.json())
    else:
        req = requests.get(URL_pose, data=params)
        # print(req.json())

    if "status" in req.json():
        return None
    else:
        return req.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Retrieve job results for CSM-AB')
    parser.add_argument('job_type', type=str, choices=['single', 'pose'],
                        help='choose between single and pose prediction')
    parser.add_argument('job_id', type=str,
                        help='Job identifier code generated upon submission')
    parser.add_argument("output_csv", type=str,
                        help="Output CSV file name")

    args = parser.parse_args()

    while not (req := main(args)):
        # throttle wait for 30 seconds
        print("Waiting for 5 seconds...")
        time.sleep(30)

    try:
        df = pd.DataFrame({'prediction': [req["prediction"]], 'typeofAb': [req["typeofAb"]]})
    except:
        import ipdb; ipdb.set_trace()

    df.to_csv(args.output_csv, index=False)
