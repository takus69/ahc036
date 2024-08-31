import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import multiprocessing
import subprocess
import json
import datetime


def run(i):
    output_str = subprocess.run(f'powershell cat in/{i:04}.txt | .\\target\\debug\\ahc036.exe > out/{i:04}.txt', shell=True, capture_output=True, text=True).stderr
    # print('output_str:', output_str.split('\n'))
    result = json.loads(output_str.split('\n')[-2])
    return result


def main(i):
    start = time.time()
    # print(i, 'start')
    r = run(i)
    t = round(time.time()-start, 4)
    M = r['M']
    LA = r['LA']
    LB = r['LB']
    score = r['score']
    lt_lb = r['lt_lb']
    path_set_len = r['path_set_len']
    pred_match_rate = r['pred_match_rate']
    actual_match_rate = r['actual_match_rate']
    data = [i, M, LA, LB, score, lt_lb, path_set_len, pred_match_rate, actual_match_rate, t]
    print('\r', 'end', i, end='')
    # print(i, 'end')
    return data


if __name__ == '__main__':
    start = time.time()
    print("start: ", datetime.datetime.fromtimestamp(start))
    trial = 200
    '''
    result = []
    for i in tqdm(range(trial)):
        data = main(i)
        result.append(data)
    '''
    processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=processes) as pool:
        data = [pool.apply_async(main, (i,)) for i in range(trial)]
        result = [d.get() for d in data]
    print()
    # '''
    df = pd.DataFrame(result, columns=['i', 'M', 'LA', 'LB', 'score', 'lt_lb', 'path_set_len', 'pred_match_rate', 'actual_match_rate', 'time'])
    score = np.mean(df['score'])
    print(f"score: {format(int(score*50), ',')}, score mean: {format(int(score), ',')}")
    df.to_csv('result.csv', index=False)
    print(f'end elapsed time: {time.time()-start:.2f}s')
