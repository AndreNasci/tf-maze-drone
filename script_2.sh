#!/bin/bash

# Combination | Gamma | Epsilon | Buffer

# Combination 0
# Variando Epsilon
#./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 0 0 0 0 0
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 0 0 1 0 0
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 0 0 2 0 0

# Variando Gamma
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 0 1 2 0 0
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 0 2 2 0 0
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 0 3 2 0 0

# Variando Buffer
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 0 2 2 1 0
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 0 2 2 2 0
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 0 2 2 3 0
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 0 2 2 4 0

# Combination 1
# Variando Epsilon
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 1 1 0 0 0
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 1 1 1 0 0
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 1 1 2 0 0

# Variando Gamma
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 1 1 2 0 0
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 1 2 2 0 0
./maze-drone-v03/bin/python 016_grid_search_3_3fases.py 1 3 2 0 0




