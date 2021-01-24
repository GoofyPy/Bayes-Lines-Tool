# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:23:54 2021

@author: Goofy
"""

# ----------------------------------
#             LIBRARIES
# ----------------------------------

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests
from datetime import date
import seaborn as sns

mpl.rcParams['font.family'] = 'Helvetica'
from matplotlib.ticker import PercentFormatter

# ----------------------------------
#         GLOBAL VARIABLES
# ----------------------------------

URL = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
FILE = 'Covid19-Dataset.csv'

SENS = np.arange(0.03, 1, 0.005)
SPEC = np.arange(0.75, 1, 0.005)
PREV = np.arange(0.001, 0.5, 0.001)

# ----------------------------------
#             FUNCTIONS
# ----------------------------------

def dl_csv(csv_url, file_name):
    # Download dataset from ourworldindata
    req = requests.get(csv_url)
    url_content = req.content
    with open(file_name, 'wb') as csv_file:
        csv_file.write(url_content)
        
def importDF(file=FILE):
    # import dataset
    df = pd.read_csv(FILE)
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    return df

def CM(sens, spec, prev, num_tests):
    # confusion matrix
    TP = sens*prev*num_tests
    FP = (1-spec)*(1-prev)*num_tests
    FN = (1-sens)*prev*num_tests
    TN = spec*(1-prev)*num_tests
    
    return TP, TN, FP, FN

def CM_test(sens, spec, prev, num_tests):
    # confusion matrix based on the formulas from the SQL code (same results as normal CM function)
    has_disease = int(prev*num_tests)
    hasnot_disease = int((1-prev)*num_tests)
    
    TP = int(sens*prev*num_tests)
    TN = int(spec*(1-prev)*num_tests)
    
    FP = hasnot_disease - TN
    FN = has_disease - TP
    
    return TP, TN, FP, FN

def getTests(df, day, location='United Kingdom'):
    # get number of tests and positive tests from dataset
    df = df[df['location']==location]
    df_day = df[df['date']==day]
    
    tests = df_day['new_tests'].values
    p_tests = df_day['new_tests'].values * df_day['positive_rate'].values
    
    return tests[0], p_tests[0]

def createCombinations(day, tests=-1, p_tests=-1):
    # iterate through all combinations and create pandas dataframe
    
    # if tests and positive tests are not specfified, use data from dataset
    if tests == -1 or p_tests == -1:
        df = importDF()
        tests, p_tests = getTests(df, day)
    
    n_rows = len(SENS)*len(SPEC)*len(PREV)
    data = np.zeros((n_rows, 9))
    
    ii = 0
    for i, prev in enumerate(PREV):
        for j, sens in enumerate(SENS):
            for k, spec in enumerate(SPEC):
                TP, TN, FP, FN = CM(sens, spec, prev, tests)
                data[ii,:] = [tests, p_tests, sens, spec, prev, TP, TN, FP, FN]
                ii += 1
                if ii % 5000 == 0:
                    print(f'{(ii+1)*100/n_rows:.2f} %')
    
    # create pandas dataframe
    df_new = pd.DataFrame(data, columns=['Tests', 'Positive Tests', 'Sensitivity', 'Specificity', 'Prevalence', 'TP', 'TN', 'FP', 'FN'])
    df_new['Positive Tests est.'] = df_new['TP'] + df_new['FP']
    df_new['Test Delta'] = df_new['Positive Tests'] - df_new['Positive Tests est.']
    df_new['Deviation'] = df_new['Test Delta'] / df_new['Positive Tests']
    
    return df_new

def checkDataFromPaper():
    # function to check if data from the paper is correct
    # values from the first 5 bars from figure 5
    tests = 536947
    p_tests = 56733
    spec = [0.91, 0.905, 0.91, 0.895, 0.905]
    sens = [0.96, 0.505, 0.67, 0.12, 0.355]
    prev = [0.02, 0.03, 0.03, 0.04, 0.04]

    print('UK 21.01.2021')
    for sp, se, pr in zip(spec, sens, prev):
        TP, TN, FP, FN = CM(se, sp, pr, tests)
        print(f'Specificity: {sp:.3f}, Sensitiviy: {se:.3f}, Prevelance: {pr:.3f}')
        print(f'Reported Positive Tests:    {p_tests}')
        print(f'Estimated Positive Tests:   {TP+FP:.0f}')
        print(f'Delta:                      {p_tests-TP-FP:.0f}')
        print(f'Deviation:                  {(p_tests-TP-FP)*100/p_tests:.2f} %\n')
        
def checkMatches(df, dev=1e-3):
    # check which combinations match the actual data 
    # matches are classified by maximum deviation between estimated positive 
    # tests and real positive tests
    df_match = df[abs(df['Deviation'])<dev]
    print(f'Combinations:            {len(df)}')
    print(f'Matches:                 {len(df_match)}')
    print(f'max. Deviation           {dev*100:.3f} %')
    
    return df_match

def plotHist(df_match):
    fig = plt.figure(figsize=(12,8))
    rgb_values = sns.color_palette("Set2", 3)
    
    spec = df_match['Specificity']
    sens = df_match['Sensitivity']
    prev = df_match['Prevalence']
    
    ax1 = plt.subplot(131)
    ax1.hist(spec, bins=50, weights=np.ones(len(spec)) / len(spec), color=rgb_values[0])
    plt.title('Specificity')
    
    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax1.grid(axis='y', alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.yaxis.set_ticks_position('none')
    ax1.tick_params(color=(0.7,0.7,0.7))
    ax1.spines['bottom'].set_color((0.7,0.7,0.7))
    
    ax2 = plt.subplot(132)
    ax2.hist(sens, bins=50, weights=np.ones(len(sens)) / len(sens), color=rgb_values[1])
    plt.title('Sensitiviy')
    
    ax2.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.grid(axis='y', alpha=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.yaxis.set_ticks_position('none')
    ax2.tick_params(color=(0.7,0.7,0.7))
    ax2.spines['bottom'].set_color((0.7,0.7,0.7))
    
    ax3 = plt.subplot(133)
    ax3.hist(prev, bins=50, weights=np.ones(len(prev)) / len(prev), color=rgb_values[2])
    plt.title('Prevalence')
    
    ax3.yaxis.set_major_formatter(PercentFormatter(1))
    ax3.grid(axis='y', alpha=0.6)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.yaxis.set_ticks_position('none')
    ax3.tick_params(color=(0.7,0.7,0.7))
    ax3.spines['bottom'].set_color((0.7,0.7,0.7))
    
    fig.tight_layout()

# ----------------------------------
#              SCRIPT
# ----------------------------------

day = date(2021, 1, 11)
df = createCombinations(day, tests=536947, p_tests = 56733)
    
df_match = checkMatches(df, 1e-2)
plotHist(df_match)

