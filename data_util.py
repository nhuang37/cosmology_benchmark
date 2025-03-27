import torch 
import numpy as np
import argparse
import pathlib
import os
import re
import pickle

def read_data(args):
    #important: read the file in order with the galaxy ID 0,1,2...
    if args.neutrinos:
        assert str(args.data_path) == '/mnt/home/fvillaescusa/public_www/Lawrence/Neutrinos/5000halos'
    all_files = os.listdir(args.data_path) 
    all_files.remove(args.y_name)
    if args.neutrinos:
        all_files.remove('create_catalogs.py')
        all_files.remove('create_catalogs.py~')
    print(len(all_files))
    #print(all_files[0])
    all_files = sorted(all_files, key=lambda x: int(re.split('[_.]', x)[1]))
    y = np.loadtxt(args.data_path / args.y_name)
    Xs = []
    for f_name in all_files:
        Xs.append(np.loadtxt(args.data_path / f_name))
    assert len(Xs) == y.shape[0], 'the number of point clouds do not match'
    assert all(X.shape == Xs[0].shape for X in Xs), 'some point clouds are not in shape (n,3)'

    n = Xs[0].shape[0]
    m = len(Xs)
    name = f'datasets/position_only/data_m={m}_n={n}_neutrinos={args.neutrinos}.pkl'
    data = {'Xs': Xs, 'y': y}
    pickle.dump(data, open(name, 'wb'))

def read_velocity_data(args):
    #important: read the file in order with the galaxy ID 0,1,2...
    all_files = os.listdir(args.data_path) 
    all_files.remove(args.y_name)
    print(len(all_files))
    #print(all_files[0])
    all_files = sorted(all_files, key=lambda x: int(re.split('[_.]', x)[1]))
    Xs, Vs = [], [] #X -> position (3d); y -> velocity (3d)
    for f_name in all_files:
        pos_vel = np.loadtxt(args.data_path / f_name)
        Xs.append(pos_vel[:,:3])
        Vs.append(pos_vel[:,3:])
    assert all(X.shape == Xs[0].shape for X in Xs), 'some position point clouds are not in shape (n,3)'
    assert all(V.shape == Vs[0].shape for V in Vs), 'some velocity point clouds are not in shape (n,3)'

    n = Xs[0].shape[0]
    m = len(Xs)
    name = f'datasets/position_velocity/position_velocity_m={m}_n={n}.pkl'
    data = {'Xs': Xs, 'y': Vs}
    pickle.dump(data, open(name, 'wb'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=pathlib.Path, \
                        default='/mnt/home/fvillaescusa/public_www/Lawrence/new', help='data directory') #/mnt/home/fvillaescusa/public_www/Lawrence/Neutrinos/5000halos
    parser.add_argument('--y_name', type=str, default='latin_hypercube_params.txt', help='y name')
    parser.add_argument('--neutrinos', action='store_true', help='use neutrino dataset')
    parser.add_argument('--velocity', action='store_true', help='use position-velocity dataset')

    args = parser.parse_args()
    if args.velocity:
        read_velocity_data(args)
    else:
        read_data(args)
