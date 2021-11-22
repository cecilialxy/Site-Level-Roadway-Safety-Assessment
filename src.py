# Import libraries
import glob
import pandas as pd
import numpy as np
import pickle
import requests
import json
import fiona
import contextily as ctx
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiPoint, Polygon

start_year,end_year = 2014,2018 # todo: read from user input

severity_color_dict = {'Fatal Crash':'black','A Injury Crash':'red','B Injury Crash':'orange', 'C Injury Crash':'green', 'No Injuries':'#4f94d4'}
time_sorter_dict = {'crashyear': np.arange(start_year,end_year+1),
                    'crashmonth': np.arange(1,13),
                    'dayofweekc': np.arange(1,8),
                   'crashhour': np.arange(0,25)}
time_label_dict = {'crashyear': 'Year',
                    'crashmonth': 'Month',
                    'dayofweekc': 'Day of week',
                   'crashhour': 'Hour'}
time_xticklabel_dict = {'crashyear': np.arange(start_year,end_year+1),
                    'crashmonth': ['Jan.','Feb.','Mar.','Apr.','May','Jun.','Jul.','Aug.','Sep.','Oct.','Nov.','Dec.'],
                    'dayofweekc': ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                   'crashhour': np.arange(0,25)}



def crash_severity_mapping(SiteCrashes):
    # This function maps crashes by severity levels
    fig, ax = plt.subplots(figsize = (15,10))
    # 150 square buffer around the intersection
    site_bounds = SiteCrashes.geometry.buffer(150, cap_style = 3).boundary.geometry.total_bounds
    xmin, ymin, xmax, ymax = site_bounds
    xlim = ([xmin-50,  xmax+50])
    ylim = ([ymin-50,  ymax+50])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Location_Rural_Intx.plot(ax=ax,color='blue')
    # SiteCrashes.plot(ax=ax,color='red')
    legend_elements = []
    scatters = []
    textplace = 0
    for inj in pd.unique(SiteCrashes['crashinjur']):
        df = SiteCrashes[SiteCrashes['crashinjur']==inj]
        df.plot(ax=ax,c=severity_color_dict[inj],markersize=50)
    ctx.add_basemap(ax, crs = "EPSG:3435", source = ctx.providers.OpenStreetMap.Mapnik)
    ax.text(0.8, 0.92, str(SiteCrashes.shape[0])+ ' crashes in total',verticalalignment='center', horizontalalignment='left',
            transform=ax.transAxes, color='black', fontsize=10)
    ax.legend(handles=legend_elements, loc=[0.8,0.8], prop={'size':20})

def counts_by_index(col,sorter,SiteCrashes):
    col_counts = pd.DataFrame(index = sorter)
    for i in sorter: # padding for 0
        if i in pd.unique(SiteCrashes[col].value_counts().index):
            col_counts.loc[i,col] = SiteCrashes[col].value_counts()[i]
        else:
            col_counts.loc[i,col] = 0
    return col_counts

def counts_by_time(col,LocationName,start_year,end_year,SiteCrashes):
    """
    This function plot the crash counts by a specific time dimension (col) and save the counts results in a csv file
    col can be one of ['crashyear','crashmonth','dayofweekc','crashhour']
    """
    SiteCrashes_target = SiteCrashes[(SiteCrashes['crashyear']>=start_year)&(SiteCrashes['crashyear']<=end_year)]
    sorter = time_sorter_dict[col]
    col_counts = counts_by_index(col,sorter,SiteCrashes_target)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.bar(col_counts.index,col_counts[col].values)
    plt.xlabel(time_label_dict[col], labelpad=20)
    plt.ylabel('Number of Crashes', labelpad=20)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xticks(sorter)
    ax.set_xticklabels(time_xticklabel_dict[col])
    plt.title('Crash Counts by '+ time_label_dict[col]+' at the '+LocationName, y=1.02)
    plt.savefig('./analysis_results/Crash Counts by '+time_label_dict[col]+'.png', dpi=600)
    col_counts.to_csv('./analysis_results/Crash Counts by '+time_label_dict[col]+'.csv',header=['counts'])

def plot_counts_by_time_statistics(start_year,end_year,SiteCrashes):
    """
    This function is a synthetic version of the counts_by_time function
    This function plots the crash counts by 4 time dimensions (cols), namely ['crashyear', 'crashmonth', 'dayofweekc', 'crashhour']
    and save the counts results in a separate csv file
    """
    fig, axs = plt.subplots(2,2, figsize = (16,10))
    SiteCrashes_target = SiteCrashes[(SiteCrashes['crashyear']>=start_year)&(SiteCrashes['crashyear']<=end_year)]
    for axes,col in zip([(0,0),(0,1),(1,0),(1,1)],['crashyear','crashmonth','dayofweekc','crashhour']):
        sorter = time_sorter_dict[col]
        col_counts = counts_by_index(col,sorter,SiteCrashes_target)
        axs[axes].bar(col_counts.index,col_counts[col].values)
        axs[axes].set_ylabel('Number of Crashes', labelpad=10)
        axs[axes].yaxis.get_major_locator().set_params(integer=True)
        axs[axes].set_xticks(sorter)
        axs[axes].set_xticklabels(time_xticklabel_dict[col])
        axs[axes].set_title('Crash Counts by '+ time_label_dict[col], y=1.02)
        col_counts.to_csv('./analysis_results/Crash Counts by '+time_label_dict[col]+'.csv',header=['counts'])

def counts_by_type(col,LocationName,start_year,end_year,SiteCrashes):
    """
    This function plot the crash counts by collision type and save the counts results in a csv file
    col: the column that describes the crash type in the dataset, 'typeoffirs'
    """
    SiteCrashes_target = SiteCrashes[(SiteCrashes['crashyear']>=start_year)&(SiteCrashes['crashyear']<=end_year)]
    sorter = SiteCrashes_target[col].value_counts().index
    col_counts = counts_by_index(col,sorter,SiteCrashes_target)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.bar(col_counts.index,col_counts[col].values)
    plt.xlabel('Type', labelpad=20)
    plt.ylabel('Number of Crashes', labelpad=20)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xticks(sorter)
    ax.set_xticklabels(sorter, rotation = 45)
    plt.title('Crash Counts by Crash Type at the '+LocationName, y=1.02)
    plt.savefig('./analysis_results/Crash Counts by Crash Type.png', bbox_inches='tight', dpi=600)
    col_counts.to_csv('./analysis_results/Crash Counts by Crash Type'+'.csv',header=['counts'])

def counts_by_type_time(col_time,col_main,LocationName,start_year,end_year,SiteCrashes):
    """
    This function plot the crash counts by collision type and time, and save the counts results in a csv file
    col_time: the column that describes the time dimension in the dataset, ['crashyear','crashmonth','dayofweekc','crashhour']
    col_main: the column that describes the crash type in the dataset, 'typeoffirs'
    """
    SiteCrashes_target = SiteCrashes[(SiteCrashes['crashyear']>=start_year)&(SiteCrashes['crashyear']<=end_year)]
    df = pd.crosstab(SiteCrashes_target[col_time],SiteCrashes_target[col_main])
    sorter = time_sorter_dict[col_time]
    for i in sorter:
        if i not in df.index:
            df.loc[i,:] = 0
    df = df.reindex(sorter)
    df.plot.bar(stacked=True)
    plt.ylabel('Number of Crashes', labelpad=20)
    plt.title('Number of Crashes by Crash Type and ' + time_label_dict[col_time]+' at the \n'+LocationName, y=1.05)
    plt.xticks(rotation=0)
    plt.legend()
    plt.legend(bbox_to_anchor=(1, 0.8))
    plt.savefig('./analysis_results/Number of Crashes by Crash Type and '+time_label_dict[col_time]+'.png', bbox_inches='tight', dpi=1000)
    df.to_csv('./analysis_results/Number of Crashes by Crash Type and '+time_label_dict[col_time]+'.csv')

def plot_type_time_statistics(col_main,LocationName,start_year,end_year,SiteCrashes):
    """
    This function is a synthetic version of the counts_by_type_time function
    This function plots the crash counts by crash type and one of the 4 time dimensions (cols), namely ['crashyear', 'crashmonth', 'dayofweekc', 'crashhour'], and save the counts results in a separate csv file
    """
    fig, axs = plt.subplots(2,2, figsize = (16,10))
    SiteCrashes_target = SiteCrashes[(SiteCrashes['crashyear']>=start_year)&(SiteCrashes['crashyear']<=end_year)]
    for axes,col_time in zip(axs.flat,['crashyear','crashmonth','dayofweekc','crashhour']):
        df = pd.crosstab(SiteCrashes_target[col_time],SiteCrashes_target[col_main])
        sorter = time_sorter_dict[col_time]
        for i in sorter:
            if i not in df.index:
                df.loc[i,:] = 0
        df = df.reindex(sorter)
        df.plot.bar(stacked=True,legend=None,ax=axes)
        df.to_csv('./analysis_results/Crash Counts by Crash Type and '+time_label_dict[col_time]+'.csv')
        axes.set_xlabel(time_label_dict[col_time])
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels,loc='upper center', bbox_to_anchor=(0.5, 0.05),
          fancybox=True, shadow=True, ncol=len(pd.unique(SiteCrashes_target[col_main])))
    fig.suptitle('Crash Counts by Crash Type and Time at the \n'+LocationName,y=0.95,fontsize=15)
    plt.savefig('./analysis_results/Number of Crashes by Crash Type and Time.png', dpi=1000)

def counts_by_time_type(col_time,col_main,LocationName,start_year,end_year,SiteCrashes):
    """
    This function plot the crash counts by time and collision type using heat map, and save the counts results in a csv file
    col_time: the column that describes the time dimension in the dataset, ['crashyear','crashmonth','dayofweekc','crashhour']
    col_main: the column that describes the crash type in the dataset, 'typeoffirs'
    """
    time = time_sorter_dict[col_time]
    SiteCrashes_target = SiteCrashes[(SiteCrashes['crashyear']>=start_year)&(SiteCrashes['crashyear']<=end_year)]
    crashtype = pd.unique(SiteCrashes_target[col_main])
    crashtype_count_time = pd.crosstab(SiteCrashes[col_time],SiteCrashes[col_main])

    ## Padding 0 counts
    if len(list(set(time) - set(crashtype_count_time.index)))>0:
        for t in list(set(time) - set(crashtype_count_time.index)):
            crashtype_count_time.loc[t,:] = 0
    ## Re-organize indices
    crashtype_count_time = crashtype_count_time.reindex(time)
    ## Re-organize by crash counts
    crashtype_count_time = crashtype_count_time[list(crashtype_count_time.sum().sort_values(ascending=False).index)]
    crashtype_count_time.to_csv('./analysis_results/Crash Counts by '+time_label_dict[col_time]+' and Crash Type.csv')
    fig = plt.subplots(figsize = (10,6))
    ticks=np.arange(crashtype_count_time.values.min(),crashtype_count_time.values.max()+1 )
    ranges = np.arange(crashtype_count_time.values.min()-0.5,crashtype_count_time.values.max()+1.5 )
    cmap = plt.get_cmap("Reds", crashtype_count_time.values.max()-crashtype_count_time.values.min()+1)
    ax = sns.heatmap(crashtype_count_time.T, annot=True, linewidths=0.4, cmap=cmap,
            cbar_kws={"ticks":ticks, "boundaries":ranges,'label': 'Crash Counts'})
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.xlabel(col_time)
    plt.ylabel(' ')
    plt.title('Crash counts by ' + time_label_dict[col_time]+' and Crash Type at the \n'+LocationName, y=1.05)
    plt.savefig('./analysis_results/Crash Type and '+col_time+'.png', bbox_inches='tight', dpi=600)

def plot_time_type_statistics(col_main,LocationName,start_year,end_year,SiteCrashes):
    """
    This function is a synthetic version of the counts_by_time_type function
    This function plots the crash counts by crash type and one of the 4 time dimensions (cols), namely ['crashyear', 'crashmonth', 'dayofweekc', 'crashhour'], and save the counts results in a separate csv file
    """
    fig, axs = plt.subplots(2,2, figsize = (20,12))
    SiteCrashes_target = SiteCrashes[(SiteCrashes['crashyear']>=start_year)&(SiteCrashes['crashyear']<=end_year)]
    for axes,col_time in zip(axs.flat,['crashyear','crashmonth','dayofweekc','crashhour']):
        time = time_sorter_dict[col_time]
        crashtype = pd.unique(SiteCrashes_target[col_main])
        crashtype_count_time = pd.crosstab(SiteCrashes[col_time],SiteCrashes[col_main])
        ## Padding 0 counts
        if len(list(set(time) - set(crashtype_count_time.index)))>0:
            for t in list(set(time) - set(crashtype_count_time.index)):
                crashtype_count_time.loc[t,:] = 0
        ## Re-organize indices
        crashtype_count_time = crashtype_count_time.reindex(time)
        ## Re-organize by crash counts
        crashtype_count_time = crashtype_count_time[list(crashtype_count_time.sum().sort_values(ascending=False).index)]
        crashtype_count_time.to_csv('./analysis_results/Crash Counts by '+time_label_dict[col_time]+' and Crash Type.csv')
        ticks=np.arange(crashtype_count_time.values.min(),crashtype_count_time.values.max()+1 )
        ranges = np.arange(crashtype_count_time.values.min()-0.5,crashtype_count_time.values.max()+1.5 )
        cmap = plt.get_cmap("Reds", crashtype_count_time.values.max()-crashtype_count_time.values.min()+1)
        sns.heatmap(crashtype_count_time.T, annot=True, linewidths=0.4, cmap=cmap,
            cbar_kws={"ticks":ticks, "boundaries":ranges,'label': 'Crash Counts'},ax=axes)
        axes.set_xlabel(time_label_dict[col_time])
        axes.set_ylabel(' ')
    fig.suptitle('Crash Counts by Time and Crash Type at the \n'+LocationName,y=0.95,fontsize=15)
    plt.savefig('./analysis_results/Number of Crashes by Time and Crash Type.png', dpi=1000)

def counts_by_severity(col,LocationName,start_year,end_year,SiteCrashes):
    """
    This function plot the crash counts by severity level and save the counts results in a csv file
    col: the column that describes the crash severity in the dataset, 'crashinjur'
    """
    SiteCrashes_target = SiteCrashes[(SiteCrashes['crashyear']>=start_year)&(SiteCrashes['crashyear']<=end_year)]
    sorter = ['Fatal Crash','A Injury Crash','B Injury Crash', 'C Injury Crash', 'No Injuries']
    col_counts = counts_by_index(col,sorter,SiteCrashes_target)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.bar(col_counts.index,col_counts[col].values)
    plt.xlabel('Severity', labelpad=20)
    plt.ylabel('Number of Crashes', labelpad=20)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xticks(sorter)
    #ax.set_xticklabels(sorter, rotation = 45)
    plt.title('Crash Counts by Severity at the '+LocationName, y=1.02)
    plt.savefig('./analysis_results/Crash Counts by Crash Severity.png', bbox_inches='tight', dpi=600)
    col_counts.to_csv('./analysis_results/Crash Counts by Crash Severity'+'.csv',header=['counts'])

def counts_by_severity_type(col_severity,col_type,LocationName,start_year,end_year,SiteCrashes):
    """
    This function plot the crash counts by severity level and collision type, and save the counts results in a csv file
    col_severity: the column that describes the crash severity, 'crashinjur'
    col_type: the column that describes the crash type in the dataset, 'typeoffirs'
    """
    severitylevel = ['Fatal Crash','A Injury Crash','B Injury Crash', 'C Injury Crash', 'No Injuries']
    SiteCrashes_target = SiteCrashes[(SiteCrashes['crashyear']>=start_year)&(SiteCrashes['crashyear']<=end_year)]
    severity_count_type = pd.crosstab(SiteCrashes_target[col_severity],SiteCrashes_target[col_type])
    ## Padding 0 counts
    if len(list(set(severitylevel) - set(severity_count_type.index)))>0:
        for s in list(set(severitylevel) - set(severity_count_type.index)):
            severity_count_type.loc[s,:] = 0
    ## Sort columns by crash counts
    severity_count_type = severity_count_type[list(severity_count_type.sum(axis=0).sort_values(ascending=False).index)]
    ## Sort indices by severity level
    severity_count_type = severity_count_type.reindex(severitylevel).astype(int)
    severity_count_type = severity_count_type.rename(index={'A Injury Crash': 'A Injury', 'B Injury Crash': 'B Injury', 'C Injury Crash': 'C Injury'})
    ## Plot
    severity_count_type.plot.bar(stacked=True)
    plt.ylabel("Number of Crashes", labelpad=20)
    plt.title('Number of Crashes by Crash Type and Severity Level at the \n'+LocationName, y=1.05)
    plt.xticks(rotation=0)
    plt.legend()
    #plt.legend(bbox_to_anchor=(1, 0.8))
    plt.savefig("./analysis_results/Number of Crashes by Severity and Crash Type.png", bbox_inches='tight', dpi=1000)

    severity_count_type['Sum'] = severity_count_type.sum(axis=1)
    severity_count_type['Percent'] = round(severity_count_type['Sum']/sum(severity_count_type['Sum']),2)
    severity_count_type.loc['Total',:] = severity_count_type.sum(axis=0)
    severity_count_type.to_csv('./analysis_results/Crash Counts by Severity Level and Crash Type.csv')

def counts_by_type_severity(col_severity,col_type,LocationName,start_year,end_year,SiteCrashes):
    """
    This function plot the crash counts by collision type and severity level, and save the counts results in a csv file
    col_severity: the column that describes the crash severity, 'crashinjur'
    col_type: the column that describes the crash type in the dataset, 'typeoffirs'
    """
    severitylevel = ['Fatal Crash','A Injury Crash','B Injury Crash', 'C Injury Crash', 'No Injuries']
    SiteCrashes_target = SiteCrashes[(SiteCrashes['crashyear']>=start_year)&(SiteCrashes['crashyear']<=end_year)]
    severity_count_type = pd.crosstab(SiteCrashes_target[col_severity],SiteCrashes_target[col_type])
    for i in severitylevel:
        if i not in severity_count_type.index:
            severity_count_type.loc[i,:] = 0
    ## sort by severity level
    severity_count_type = pd.DataFrame.transpose(severity_count_type.reindex(severitylevel))
    ## sort by crash counts
    severity_count_type = severity_count_type.reindex(severity_count_type.sum(axis=1).sort_values(ascending=False).index)
    ## Plot
    severity_count_type.plot.bar(stacked=True,color=list({key: severity_color_dict[key] for key in severitylevel}.values()))
    plt.xlabel(' ')
    plt.ylabel("Number of Crashes", labelpad=20)
    plt.xticks(rotation=30)
    plt.legend(title="Severity Level")
    plt.title('Number of Crashes by Severity Level and Crash Type at the \n'+LocationName, y=1.05)
    plt.savefig("./analysis_results/Number of Crashes by Severity Level and Crash Type.png", bbox_inches='tight', dpi=600)

    severity_count_type['Sum'] = severity_count_type.sum(axis=1)
    severity_count_type['Percent'] = round(severity_count_type['Sum']/sum(severity_count_type['Sum']),2)
    severity_count_type.loc['Total',:] = severity_count_type.sum(axis=0)
    severity_count_type.to_csv('./analysis_results/Crash Counts by Crash Type and Severity Level.csv')

def plot_time_severity_statistics(col_main,LocationName,start_year,end_year,SiteCrashes):
    """
    This function plots the crash counts by crash severity and one of the 4 time dimensions (cols), namely ['crashyear', 'crashmonth', 'dayofweekc', 'crashhour'], and save the counts results in a separate csv file
    """
    fig, axs = plt.subplots(2,2, figsize = (20,12))
    SiteCrashes_target = SiteCrashes[(SiteCrashes['crashyear']>=start_year)&(SiteCrashes['crashyear']<=end_year)]
    severitylevel = ['Fatal Crash','A Injury Crash','B Injury Crash', 'C Injury Crash', 'No Injuries']
    for axes,col_time in zip(axs.flat,['crashyear','crashmonth','dayofweekc','crashhour']):
        time = time_sorter_dict[col_time]
        crashtype = pd.unique(SiteCrashes_target[col_main])
        crashseverity_count_time = pd.crosstab(SiteCrashes[col_time],SiteCrashes[col_main])
        ## Padding 0 counts for time
        if len(list(set(time) - set(crashseverity_count_time.index)))>0:
            for t in list(set(time) - set(crashseverity_count_time.index)):
                crashseverity_count_time.loc[t,:] = 0
        ## Re-organize indices
        crashseverity_count_time = crashseverity_count_time.reindex(time)
        ## Padding 0 counts for severity level
        if len(crashseverity_count_time.columns) < len(severitylevel):
            for s in list(set(severitylevel) - set(crashseverity_count_time.columns)):
                crashseverity_count_time.loc[:,s] = 0
        ## Re-organize by severity level
        crashseverity_count_time = crashseverity_count_time[severitylevel]
        crashseverity_count_time.to_csv('./analysis_results/Crash Counts by '+time_label_dict[col_time]+' and Severity Level.csv')
        ticks=np.arange(crashseverity_count_time.values.min(),crashseverity_count_time.values.max()+1 )
        ranges = np.arange(crashseverity_count_time.values.min()-0.5,crashseverity_count_time.values.max()+1.5 )
        cmap = plt.get_cmap("Reds", crashseverity_count_time.values.max()-crashseverity_count_time.values.min()+1)
        sns.heatmap(crashseverity_count_time.T, annot=True, linewidths=0.4, cmap=cmap,
            cbar_kws={"ticks":ticks, "boundaries":ranges,'label': 'Crash Counts'},ax=axes)
        axes.set_xlabel(time_label_dict[col_time])
        axes.set_ylabel(' ')
    fig.suptitle('Crash Counts by Time and Severity Level at the \n'+LocationName,y=0.95,fontsize=15)
    plt.savefig('./analysis_results/Number of Crashes by Time and Severity Level.png', dpi=1000)

# Moduel: Pedestrian and bicyclists
def plot_ped_bicy_per_direction(LocationName,Total_Volume_Class_Breakdown,leg,direction,color,start_time,end_time):
    # The function plots the number of pedestrians and bicyclists on the specific leg per direction
    fig, ax = plt.subplots(figsize = (15,6))
    width = 0.4
    start_idx = Total_Volume_Class_Breakdown.index.get_loc(start_time)
    end_idx = Total_Volume_Class_Breakdown.index.get_loc(end_time)
    Leg_Peds = Total_Volume_Class_Breakdown.iloc[start_idx:end_idx+1,:][[(leg,direction,'Peds CW'),(leg,direction, 'Peds CCW')]]
    plt.bar(np.arange(len(Leg_Peds.index))-width, Leg_Peds[(leg,direction,'Peds CW')], align='center',width=width, color=color)
    plt.bar(np.arange(len(Leg_Peds.index)), Leg_Peds[(leg,direction,'Peds CCW')], align='center',width=width, color=color,hatch='/',alpha=0.5)
    plt.xlabel('Time',fontsize=15)
    plt.xticks(np.arange(len(Leg_Peds.index))-width/2,Leg_Peds.index,rotation=90)
    plt.ylabel('Traffic Counts',fontsize=15)
    ymax = max(Total_Volume_Class_Breakdown.iloc[start_idx:end_idx+1,:].xs('Peds CW',axis=1, level=2, drop_level=False).values.max(),
        Total_Volume_Class_Breakdown.iloc[start_idx:end_idx+1,:].xs('Peds CCW',axis=1, level=2, drop_level=False).values.max())
    ylim = ([0,  ymax+1])
    ax.set_ylim(ylim)
    plt.legend(['Clockwise','Counter Clockwise'])
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.1)
    plt.grid(axis='y')
    plt.title('Pedestrians and Bicyclists Volume at the \n'+'on '+LocationName+leg+'Leg ', y=1.05,fontsize=15)
    plt.savefig('./analysis_results/Pedestrians and Bicyclists Volume at the'+LocationName+leg+'Leg .png', bbox_inches='tight', dpi=600)
