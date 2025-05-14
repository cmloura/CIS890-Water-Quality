import numpy as np
import pandas as pd
import os
import sys
import statistics
#import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy import stats

data_path = "D:/Misc/hw3-ai/bad_chemicals.txt"
chemical = 'Nitrite'
def statistical_analysis_per_year(path):
    failed_bow = []
    date_dict = {}
    with open(path, 'r') as rfile:
        for line in rfile:
            try:
                tokens = line.strip().split(',')
                if tokens[3] != chemical.lower():
                    continue
                else:
                    if tokens[0] not in failed_bow:
                        failed_bow.append(tokens[0])
                    date_tokens =  tokens[4].split('/')
                    if date_tokens[2] not in date_dict:
                        date_dict[date_tokens[2]] = [float(tokens[1])]
                    else:
                        date_dict[date_tokens[2]].append(float(tokens[1]))
            except Exception:
                continue

    rfile.close()


    sorted_d = dict(sorted(date_dict.items()))
    xs = list(sorted_d.keys())
    ys = []
    print('Number of bodies of water that failed chemical test: {}'.format(len(failed_bow)))
    for k,v in sorted_d.items():
        ys.append(sum(v)/len(v))
        print(f"Date: {k}\tAverage: {sum(v)/len(v)}\tStandard Deviation: {statistics.stdev(v)}")

    slope, intercept, r, p, std_err = stats.linregress(xs, ys)
    plt.xlabel("Date")
    plt.ylabel(f"Average {chemical} Level (mg/L)")
    plt.title(f'Time Series Analysis of {chemical} Levels')
    plt.plot(xs, ys, marker='o', linestyle='-')
    plt.plot(slope*xs + intercept, marker='o', color='#00000', linestyle='-')
    plt.xticks(rotation=45)
    plt.grid(True)


    plt.show()

def cluster(filepath):
    column_names = ["Water Body", "Measurement", "Unit", "Chemical", "Date"]
    df = pd.read_csv(filepath, names=column_names, parse_dates=["Date"])

    df["Measurement"] = pd.to_numeric(df["Measurement"], errors="coerce")
    pivot_df = df.pivot_table(index="Water Body", columns="Chemical", values="Measurement", aggfunc="mean").fillna(0)
    plt.figure(figsize=(12, 6))
    linkage_matrix = linkage(pivot_df, method='ward')

    dendrogram(linkage_matrix, labels=pivot_df.index, leaf_rotation=90, leaf_font_size=10)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Bodies of Water")
    plt.ylabel("Cluster Distance")
    plt.show()

    plt.figure(figsize=(12,8))
    sns.clustermap(pivot_df, method='ward', cmap='coolwarm', standard_scale=1, figsize=(10,10))
    plt.title('Heatmap of Chemical Concentration')
    plt.show()

def cluster_one_year(filepath):
    yearly_cluster_res = {}

    column_names = ["Water Body", "Measurement", "Unit", "Chemical", "Date"]
    df = pd.read_csv(filepath, names=column_names, parse_dates=["Date"])
    df["Measurement"] = pd.to_numeric(df["Measurement"], errors="coerce")

    df['Year'] = df['Date'].dt.year

    for year in df['Year'].unique():
        yearly_df = df[df['Year'] == year]

        pivot_df = yearly_df.pivot_table(index='Water Body', columns = 'Chemical', values='Measurement', aggfunc='mean').fillna(0)
        yearly_cluster_res[year] = pivot_df
        plt.figure(figsize=(12, 6))
        linkage_matrix = linkage(pivot_df, method='ward')
        dendrogram(linkage_matrix, labels=pivot_df.index, leaf_rotation=90, leaf_font_size=10)
        plt.title(f'Hierarchical Clustering for {year}')
        plt.xlabel('Bodies of Water')
        plt.ylabel('Cluster Distance')
        plt.show()

        plt.figure(figsize=(12, 8))
        sns.clustermap(pivot_df, method='ward', cmap='coolwarm', standard_scale=1, figsize=(10,10))
        plt.title(f'Heatmap of Chemical Concentrations for {year}')
        plt.show()



def parse_chems(txtfilepath):
    try:
        df = pd.read_csv(txtfilepath, names=['Water Body', 'Measurement', 'Unit', 'Chemical', 'Date'], parse_dates=['Date'])
        return df
    except Exception as e:
        print(f"Error reading {txtfilepath}: {e}")
        return None

def create_limit_dict(limits_path):
    dick = {}
    with open(limits_path, 'r') as limits:
        for line in limits:
            tokens = line.split(',')
            dick[tokens[0]] = float(tokens[1])
    limits.close()
    return dick

    
def time_series_analysis(data_path):
    literally_everything = []

    for waterbody in os.listdir(data_path):
        waterpath = os.path.join(data_path, waterbody)
        if not os.path.isdir(waterpath):
            continue
        
        for chemical in os.listdir(waterpath):
            chempath = os.path.join(waterpath, chemical)
            if not os.path.isdir(chempath):
                continue

            for txtfile in os.listdir(chempath):
                if txtfile.endswith('.txt'):
                    txtfilepath = os.path.join(chempath, txtfile)
                    df = parse_chems(data_path)

    if df is not None and not df.empty:
        #df['Water Body'] = waterbody
        #df['Chemical'] = chemical
        literally_everything.append(df)
    fulldata = pd.concat(literally_everything, ignore_index=True)

    fulldata['Date'] = pd.to_datetime(fulldata['Date'])
    fulldata.sort_values(by=['Date'], inplace=True)

    for chemical in fulldata['Chemical'].unique():
        chemdata = fulldata[fulldata['Chemical'] == chemical]

        plt.figure(figsize=(10,5))
        for water_body in chemdata["Water Body"].unique():
            subset = chemdata[chemdata["Water Body"] == water_body]
            plt.plot(subset["Date"], subset["Measurement"], marker='o', linestyle='-', label=water_body)
        
        plt.xlabel("Date")
        plt.ylabel(f"Measurement")
        plt.title(f'Time Series Analysis of {chemical} Levels')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        if(len(sys.argv) > 1 and sys.argv[1].lower() == 'save'):
            os.makedirs('saved-images', exist_ok=True)
            filename = "saved-images/{}-time-series-analysis-graph.png".format(chemical)
            plt.savefig(filename)
            print("Graph saved as {}".format(filename))
        else:
            plt.show()

def convert(fileloc):
    with open(fileloc, 'r') as rfile:
        with open('better_limits.txt', 'w') as wfile:
            for line in rfile:
                tokens = line.strip().split(',')
                if tokens[2] == 'ug/L':
                    tokens[1] = str(float(tokens[1]) * 0.001)
                    tokens[2] = 'mg/L'
                wfile.write(",".join(tokens))
                wfile.write("\n")

def all_failed_tests(filepath, limit_dict):
    with open(filepath, 'r') as rfile:
        with open('new_bad_chemicals.txt', 'w') as wfile:
            for line in rfile:
                tokens = line.strip().split(',')
                if tokens[4] in limit_dict:
                    if float(tokens[5]) > float(limit_dict[tokens[4]]):
                        wfile.write(line)
    
        wfile.close()
    rfile.close()

def rewrite_limits_again(filepath):
    with open(filepath, 'r') as limitsfile:
        with open("chemical_num.csv", 'r') as chemical_nums:
            with open("finalized_limits.txt", 'w') as wfile:
                for line in limitsfile:
                    tokens = line.strip().split(',')
                    for line2 in chemical_nums:
                        tokens2 = line2.strip().split(',')
                        if tokens[0].lower() == tokens2[0].lower():
                            tokens[0] = tokens2[2]
                            wfile.write(",".join(tokens))
                            break

#rewrite_limits_again('better_limits.txt')
#all_failed_tests("renamed_has_socio.csv", create_limit_dict('better_limits.txt'))

def get_count(filepath):
    count_dict = {}
    with open(filepath, 'r') as rfile:
        next(rfile)
        for line in rfile:
            tokens = line.strip().split(',')
            if tokens[4] not in count_dict:
                count_dict[tokens[4]] = 1
            else:
                count_dict[tokens[4]] += 1
    rfile.close()
    return count_dict
    
def main():
    count_dict = get_count("new_bad_chemicals.txt")
    for k, v in count_dict.items():
        print("{}: {}".format(k, v))

main()