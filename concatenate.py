import pandas as pd
from os import listdir
from os.path import isfile, join

def main():
    # read_file = pd.read_csv("logs/01-rewards-combinations/03_comb-1-run-1.csv")
    # df_full = pd.DataFrame(read_file)

    # df_base = df_full.copy()

    # df_base['Average Return'] += df_full['Average Return']

    # print(df_base.head())
    # print(df_full.head())

    # df_base['Average Return'] /= 2.

    # print(df_base.head())

    # Path dos arquivos a serem listados
    path = "/home/naski/Documents/dev/maze_drone_v02/logs/01-rewards-combinations"

    files = [f for f in listdir(path) if isfile(join(path, f))]
    lista_csv = [arq for arq in files if arq.lower().endswith(".csv")]

    comb = 4
    comb_files = []

    for file in lista_csv:
        #print(file)
        
        if(int(file[8]) == comb):
            comb_files.append(file)

    print("Realizando a média dos seguintes arquivos:")
    for file in comb_files:
        print(file)

    n_files = len(comb_files)

    df_base = pd.read_csv(join(path, comb_files[0]))
    comb_files.pop(0)

    for file in comb_files:
        #print(df_base.head())
        df_to_add = pd.read_csv(join(path, file))
        #print(df_to_add.head())
        df_base += df_to_add

    print("Calculando a média...")
    df_base /= n_files

    print(df_base.head())

    df_base.to_csv(f"logs/01-rewards-combinations/Average/{file[:10]}avg.csv", index=None, header=True)

main()