import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import sys


def main():
    if len(sys.argv) != 5:
        #print("Args: XX Y -> XX = file's 2 first digits, Y -> combination, Z -> run")
        print("Args: XX Y Z -> XX = file's 2 first digits, Y -> combination, Z -> run")
        return -1
    
    file_num = sys.argv[1]
    arg_1 = int(sys.argv[2])
    arg_2 = int(sys.argv[3])
    arg_3 = int(sys.argv[4])
    

    try:
        #read_file = pd.read_csv(f"logs/06-hist-env/{file_num}-checkpoint_0-run_1.csv")
        if file_num == "006":
            read_file = pd.read_csv(f"logs/06-hist-env/{file_num}-combination_{arg_1}-epsilon_{arg_2}-gamma_{arg_3}.csv")
        elif file_num == "007":
            read_file = pd.read_csv(f"logs/06-hist-env/{file_num}-combination_0-epsilon_{arg_1}-gamma_{arg_2}-buffer_{arg_3}.csv")
        else:
            read_file = pd.read_csv(f"logs/06-hist-env/{file_num}-combination_{arg_1}-buffer_{arg_2}-fc_{arg_3}.csv")
    except FileNotFoundError: 
        print(f"\nErro: Arquivo {file_num}_comb-{arg_1}-run-{arg_2}.csv nao encontrado.")
        print("\nFinalizando...")
        sys.exit()
    except Exception:
        print(Exception)
        print("\nFinalizando...")
        sys.exit()
    

    # Read the file and stores data in a dataframe
    df = pd.DataFrame(read_file)

    num_lines, _ = df.shape
    period = int(num_lines * 0.3)

    print(f"Média das métricas nos últimos {period} episódios:")
    print(df[-period:].mean())


    # plot_all(1, "Average Return", df['Average Return'])
    # plot_all(2, "% Finished", df['% Finished'])
    # plot_all(3, "Crash Counter", df['Crash Counter'])
    # plot_all(4, "Stuck Counter", df['Stuck Counter'])
    # plot_all(5, "Avg Steps/Episode", df['Avg Steps/Episode'])
    # plot_all(6, "Loss", df['Loss log'])



if __name__ == "__main__":
    main()