import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

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

def main():
    read_file = pd.read_csv("logs/01-rewards-combinations/Average/03_comb-4-avg.csv")
    df = pd.DataFrame(read_file)

    plot_moving_avg("Average Return", df['Average Return'])
    plot_moving_avg("% Finished", df['% Finished'])
    plot_moving_avg("Crash Counter", df['Crash Counter'])
    plot_moving_avg("Stuck Counter", df['Stuck Counter'])
    plot_moving_avg("Avg Steps/Episode", df['Avg Steps/Episode'])
    

main()