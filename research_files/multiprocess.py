from multiprocessing import Pool
from continuous_test import main

def run_multiple():
    import argparse
    import sys
    from copy import deepcopy

    # Setup the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='YAML-file')
    parser.add_argument('-s', '--steps', help='Number of steps', type=int, default=500)
    parser.add_argument('-o', '--outfile', help='File for measurements', default='measurements')
    parser.add_argument('-p', '--pedestrian_file', help='Pedestrian file', default='')
    parser.add_argument('-m', '--max_pedestrians', help='max pedestrians', type=int, default=100)
    parser.add_argument('-r', '--repetitions', default=1, type=int)
    parser.add_argument('-b', '--barriertime', default=0, type=int)

    # Get basic parser object
    file = "situations/turbulence_parameters_normal.yaml"
    args = parser.parse_args([file])

    # Set default parameters
    args.steps = 30000

    # nums = range(300, 510, 20)
    # barriers = [8, 16, 24]
    # repetitions = range(1, 6)

    nums = [300]
    barriers = [8]
    repetitions = range(1, 6)

    arg_list = []

    for m in nums:
        args.max_pedestrians = m
        for b in barriers:
            args.barriertime = b
            for r in repetitions:
                args.outfile = "tbn_{}_{}_{}".format(m, b, r)
                args.pedestrian_file = "stable/tbn_{}_{}.pickle".format(m, r)
                arg_list.append(deepcopy(args))
    
    p = Pool(5)
    p.map(main, arg_list)

if __name__ == '__main__':
    run_multiple()