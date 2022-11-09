# Commit 6fe477 이전에 generate.py에서 생성된 multi-gpu json 파일, 또는
# 여러 Fold를 합쳐 만든 json 파일들에 'categories' 필드에 폴드 하나짜리의 json이 통째로 들어가있는
# 버그가 발생한다.
# 본 도구는 이 버그로 인하여 잘못된 'categories' 필드를 가지고 있는 json 파일들을 읽고 fix 하는 도구이다.
import glob
from argparse import ArgumentParser
import os
import json
from tqdm.auto import tqdm
from pathlib import Path
import shutil


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('location', help="Folder to find json files")
    args = parser.parse_args()

    json_file_list = glob.glob(os.path.join(args.location, '**', '*.json'), recursive=True)
    
    for idx, json_file in enumerate(json_file_list):
        print("({}/{}) Loading file {} ...".format(idx + 1, len(json_file_list), json_file))
        with open(json_file) as f:
            body = json.load(f)
        
        # Check if categories has certain fields or not
        if type(body['categories']) == type({}):
            print("Found buggy categories on file {}!".format(json_file))
            body['categories'] = body['categories']['categories']

            # backup_path = os.path.join(Path(json_file).parent, Path(json_file).stem + '.bak')
            # shutil.move(json_file, backup_path)
            # print("Backed up into {}!".format(backup_path))

            # print("Writting file {}!".format(json_file))
            # with open(json_file) as f:
            #     json.dump(body, f)

    print("Done")