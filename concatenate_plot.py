import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import sys

def plot_moving_avg(y_name, y_data, period=10, ylim=False, top_lim=0, bot_lim=0):
    y_axis_df = pd.DataFrame({y_name: y_data})
    
    y_axis_df['moving_avg'] = y_axis_df[y_name].rolling(window=period).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(y_axis_df[y_name], label='Original Values', marker='o')
    plt.plot(y_axis_df['moving_avg'], label=f'Moving Average ({period} periods)', linestyle='dashed')
    plt.xlabel('Period')
    plt.ylabel(y_name)
    plt.title(f'Moving average - {y_name}')
    plt.legend()
    plt.grid(axis='y')
    if ylim:
        plt.ylim(top=top_lim, bottom=bot_lim)
    plt.show()



def plot_all(plt_ax, y_name, y_data, period=50, ylim=False, top_lim=0, bot_lim=0):
    
    y_axis_df = pd.DataFrame({y_name: y_data})
    y_axis_df['moving_avg'] = y_axis_df[y_name].rolling(window=period).mean()

    plt.subplot(1, 6, plt_ax)

    plt.plot(y_axis_df[y_name], label='Original Values', marker='o')
    plt.plot(y_axis_df['moving_avg'], label=f'Mov. Avg ({period} periods)', linestyle='dashed')
    plt.xlabel('Period')
    plt.ylabel(y_name)
    plt.title(f'Moving average - {y_name}')
    plt.legend()
    plt.grid(axis='y')

    if ylim:
        plt_ax.ylim(top=top_lim, bottom=bot_lim)


def main():
    if len(sys.argv) != 4:
        print("Args: XX Y -> XX = file's 2 first digits, Y -> combination, Z -> run")
        return -1
    
    file_num = sys.argv[1]
    comb = int(sys.argv[2])
    run = int(sys.argv[3])
    

    #read_file = pd.read_csv(f"logs/01-rewards-combinations/Average/{file_num}_comb-{comb}-avg.csv")
    #read_file = pd.read_csv(f"logs/02-stuck-improving/Average/{file_num}_comb-{comb}-avg.csv")
    #read_file = pd.read_csv(f"logs/02-stuck-improving/{file_num}_comb-{comb}-avg.csv")
    #read_file = pd.read_csv(f"logs/03-walls/{file_num}_comb-{comb}-run-{run}.csv")
    try:
        read_file = pd.read_csv(f"logs/04-stateChange/{file_num}_comb-{comb}-run-{run}.csv")
    except FileNotFoundError: 
        print(f"\nErro: Arquivo {file_num}_comb-{comb}-run-{run}.csv nao encontrado.")
        print("\nFinalizando...")
        sys.exit()
    except Exception:
        print(Exception)
        print("\nFinalizando...")
        sys.exit()
    


    df = pd.DataFrame(read_file)

    # plot_moving_avg("Average Return", df['Average Return'])
    # plot_moving_avg("% Finished", df['% Finished'])
    # plot_moving_avg("Crash Counter", df['Crash Counter'])
    # plot_moving_avg("Stuck Counter", df['Stuck Counter'])
    # plot_moving_avg("Avg Steps/Episode", df['Avg Steps/Episode'])

    # Create subplots
    plt.figure(figsize=(20, 3))
    plot_all(1, "Average Return", df['Average Return'])
    plot_all(2, "% Finished", df['% Finished'])
    plot_all(3, "Crash Counter", df['Crash Counter'])
    plot_all(4, "Stuck Counter", df['Stuck Counter'])
    plot_all(5, "Avg Steps/Episode", df['Avg Steps/Episode'])
    plot_all(6, "Loss", df['Loss log'])

    # Ajustar o layout
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()