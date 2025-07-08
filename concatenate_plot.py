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



def plot_all(plt_ax, y_name, y_data, period=-1, ylim=False, top_lim=0, bot_lim=0, phase_change=0):

    
    y_axis_df = pd.DataFrame({y_name: y_data})

    # Calculates an appropriate value of period
    if period == -1:
        num_linhas, _ = y_axis_df.shape
        period = int(num_linhas * 0.3)

    y_axis_df['moving_avg'] = y_axis_df[y_name].rolling(window=period).mean()

    plt.subplot(1, 6, plt_ax)

    plt.plot(y_axis_df[y_name], label='Original Values')
#    plt.plot(y_axis_df[y_name], label='Original Values', marker='o')
    plt.plot(y_axis_df['moving_avg'], label=f'Mov. Avg ({period} periods)', linestyle='dashed')
    plt.xlabel('Period (x100 steps)')
    plt.ylabel(y_name)
    plt.title(f'Moving average - {y_name}')
    plt.legend()
    plt.grid(axis='y')

    
    # Vertical line (change of phases)
    if phase_change:
        plt.axvline(x=phase_change, color='red', linestyle='dotted', linewidth=1)

    if ylim:
        plt_ax.ylim(top=top_lim, bottom=bot_lim)


def main():
    if len(sys.argv) != 6:
        #print("Args: XX Y -> XX = file's 2 first digits, Y -> combination, Z -> run")
        print("Args: XX Y Z -> XX = file's 2 first digits, Y -> combination, Z -> run")
        return -1
    
    file_num = sys.argv[1]
    arg_1 = int(sys.argv[2])
    arg_2 = int(sys.argv[3])
    arg_3 = int(sys.argv[4])
    arg_4 = int(sys.argv[5])
    

    #read_file = pd.read_csv(f"logs/01-rewards-combinations/Average/{file_num}_comb-{comb}-avg.csv")
    #read_file = pd.read_csv(f"logs/02-stuck-improving/Average/{file_num}_comb-{comb}-avg.csv")
    #read_file = pd.read_csv(f"logs/02-stuck-improving/{file_num}_comb-{comb}-avg.csv")
    #read_file = pd.read_csv(f"logs/03-walls/{file_num}_comb-{comb}-run-{run}.csv")
    try:
        #read_file = pd.read_csv(f"logs/04-stateChange/{file_num}_comb-{comb}-run-{run}.csv")
        #read_file = pd.read_csv(f"logs/06-hist-env/{file_num}-checkpoint_0-run_1.csv")
        # read_file = pd.read_csv(f"logs/06-hist-env/011-combination_0-epsilon_2-gamma_1-buffer_0-lr_1.csv")
        if file_num == "007":
            read_file = pd.read_csv(f"logs/06-hist-env/{file_num}-combination_0-epsilon_{arg_1}-gamma_{arg_2}-buffer_{arg_3}.csv")
        elif file_num == "010" or file_num == "014" or file_num == "015" or file_num == "017":
            read_file = pd.read_csv(f"logs/06-hist-env/{file_num}-combination_0-epsilon_{arg_1}-gamma_{arg_2}-buffer_{arg_3}-lr_1.csv")
        elif file_num == "012" or file_num == "013":
            read_file = pd.read_csv(f"logs/06-hist-env/{file_num}-combination_0-epsilon_{arg_1}-gamma_{arg_2}-buffer_{arg_3}-lr_1.csv")
        else:
            read_file = pd.read_csv(f"logs/06-hist-env/{file_num}-combination_{arg_1}-epsilon_{arg_2}-gamma_{arg_3}.csv")
    except FileNotFoundError: 
        print(f"\nErro: Arquivo {file_num}_comb-{arg_1}-run-{arg_2}.csv nao encontrado.")
        print("\nFinalizando...")
        sys.exit()
    except Exception:
        print(Exception)
        print("\nFinalizando...")
        sys.exit()
    


    df = pd.DataFrame(read_file)
    #df = df[:100]
    # plot_moving_avg("Average Return", df['Average Return'])
    # plot_moving_avg("% Finished", df['% Finished'])
    # plot_moving_avg("Crash Counter", df['Crash Counter'])
    # plot_moving_avg("Stuck Counter", df['Stuck Counter'])
    # plot_moving_avg("Avg Steps/Episode", df['Avg Steps/Episode'])

    # Create subplots
    plt.figure(figsize=(20, 3))
    plot_all(1, "Average Return", df['Average Return'], phase_change=arg_4)
    plot_all(2, "% Finished", df['% Finished'], phase_change=arg_4)
    plot_all(3, "Crash Counter", df['Crash Counter'], phase_change=arg_4)
    plot_all(4, "Stuck Counter", df['Stuck Counter'], phase_change=arg_4)
    plot_all(5, "Avg Steps/Episode", df['Avg Steps/Episode'], phase_change=arg_4)
    plot_all(6, "Loss", df['Loss log'], phase_change=arg_4)

    # Ajustar o layout
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()