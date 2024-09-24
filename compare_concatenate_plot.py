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



def plot_all(plt_ax, quantidade_imagens, y_name, y_data, period=10, ylim=False, top_lim=0, bot_lim=0):
    
    y_axis_df = pd.DataFrame({y_name: y_data})
    y_axis_df['moving_avg'] = y_axis_df[y_name].rolling(window=period).mean()

    plt.subplot(quantidade_imagens, 5, plt_ax)

    plt.plot(y_axis_df[y_name], label='Original Values', marker='o')
    plt.plot(y_axis_df['moving_avg'], label=f'Moving Average ({period} periods)', linestyle='dashed')
    plt.xlabel('Period')
    plt.ylabel(y_name)
    plt.title(f'Moving average - {y_name}')
    plt.legend()
    plt.grid(axis='y')

    if ylim:
        plt_ax.ylim(top=top_lim, bottom=bot_lim)


def main():
    if len(sys.argv) % 2 != 0 or len(sys.argv) < 6:
        print("Argument error.")
        print("Usage: Z [XX Y] -> Z = number of comparatives, XX = file's first digits, Y = combination")
        return -1
    
    file_num = []
    comb = []

    # Quantidade args com exceção do nome do arquivo
    quantidade_args = len(sys.argv)

    # Criar array de file_digits e combinations
    for i in range(2, quantidade_args, 2):
    
        file_num.append(sys.argv[i])
        comb.append(int(sys.argv[i+1]))
    

    # Create subplots
    quantidade_imagens = int((quantidade_args-1)/2)
    plt.figure(figsize=(20, 3 * quantidade_imagens))

    for i in range(1, quantidade_imagens+1):
    
        read_file = pd.read_csv(f"logs/01-rewards-combinations/Average/{file_num[i-1]}_comb-{comb[i-1]}-avg.csv")
        df = pd.DataFrame(read_file)


        plot_all(1 + 5 * (i-1), quantidade_imagens, "Average Return", df['Average Return'])
        plot_all(2 + 5 * (i-1), quantidade_imagens, "% Finished", df['% Finished'])
        plot_all(3 + 5 * (i-1), quantidade_imagens, "Crash Counter", df['Crash Counter'])
        plot_all(4 + 5 * (i-1), quantidade_imagens, "Stuck Counter", df['Stuck Counter'])
        plot_all(5 + 5 * (i-1), quantidade_imagens, "Avg Steps/Episode", df['Avg Steps/Episode'])


    # Ajustar o layout
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()