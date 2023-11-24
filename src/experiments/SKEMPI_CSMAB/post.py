import argparse
import requests
import sys

URL_single = "https://biosig.lab.uq.edu.au/csm_ab/api/prediction_single"
URL_pose = "https://biosig.lab.uq.edu.au/csm_ab/api/prediction_pose"


def main(args):
    if args.job_type == 'single':
        pdb_file = args.pdb_file
        pdb_accession = args.pdb_accession
        if pdb_accession:
            data = {"pdb_accession": pdb_accession}
            req = requests.post(URL_single, data=data)
            print(req.json()["job_id"])
        else:
            pdb_to_submit = {"pdb_file": pdb_file}
            req = requests.post(URL_single, files=pdb_to_submit)
            try:
                print(req.json()["job_id"])
            except:
                print(req.json())
                raise Exception("Error in submission")
        return True
    else:
        receptor_file = args.receptor_file
        pose_file = args.pose_file
        file_to_submit = {
            'receptor_file': receptor_file, 'pose_file': pose_file}
        req = requests.post(URL_pose, files=file_to_submit)
        print(req.json()["job_id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Job submission for CSM-AB')
    parser.add_argument('job_type', type=str, choices=['single', 'pose'],
                        help='choose between single and pose prediction')
    parser.add_argument('--pdb_file', dest='pdb_file',
                        type=argparse.FileType('r'), help='PDB file')
    parser.add_argument('--pdb_accession', dest='pdb_accession',
                        type=str, help='Type PDB-ID')
    parser.add_argument('--receptor_file', dest='receptor_file',
                        type=argparse.FileType('r'), help='PDB file')
    parser.add_argument('--pose_file', dest='pose_file',
                        type=argparse.FileType('r'), help='PDB file')

    args = parser.parse_args()
    # print(args.job_type, args.pdb_file, args.pdb_accession)
    if args.job_type == 'single':
        if (args.pdb_file == None and args.pdb_accession == None):
            print(
                'post.py: error: missing arguments: Please provide --pdb_file or --pdb_accession')
            sys.exit(1)
    else:
        if args.receptor_file == None or args.pose_file == None:
            print(
                'post.py: error: missing arguments: Please provide --receptor_file or --pose_file')
            sys.exit(1)
    main(args)
