#Step 1 import required modules
import pandas as pd
import numpy as np
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from IPython.display import display
%matplotlib inline

##Step 2 Read Data

#Step 3 Create Target Variable
b = np.where((df_merged.m_u_middle_gap_indicator== 1) | (df_merged.m_d_middle_gap_indicator==1))[0]
df_merged['middle_gap_combined'] = 0
df_merged['middle_gap_combined'].iloc[(b)] = 1

# Step 4 Check stats of data 
print df_merged.describe()

#Step 5 Drop all constant value columns

constant_value_columns = np.where(df_merged.nunique()==1)
constant_value_columns = [constant_value_columns[0]]
df_merged.drop(df_merged.columns[constant_value_columns],axis=1,inplace=True)

#Step 6 Drop all columns with has missing value more than 70%.
null_value_columns = np.where((df_merged.isnull().sum())/(df_merged.shape[0]) > 0.7)
null_value_columns = [null_value_columns[0]]
df_merged = df_merged.drop(df_merged.columns[null_value_columns],axis=1)

#Step 7 Change Data Type of columns 
numeric_originally_object = ['s1_u_upload_data_volume', 's1_u_download_data_volume', 'rsrp', 'rsrq','n1_pci', 'n1_rsrp', 'n1_rsrq'
                            , 'ta_distance', 'intersite_distance', 'ue_power_headroom', 'pusch_sinr', 'pucch_sinr','rlc_pdu_download_volume','rlc_pdu_upload_volume']


# Categorical columns that should have been numerical
for col in numeric_originally_object:
    df_merged[col] = df_merged[col].apply(pd.to_numeric, args=('coerce',))
    

# Numerical Columns that should have been Categorical

object_originally_numeric = ['c_failed', 'c_error_code', 'c_call_completed', 'm_u_call_completed', 'm_u_destination_port'
                            ,'m_u_long_call_indicator', 'm_u_middle_gap_indicator','m_u_short_call_indicator'
                            ,'m_u_one_way_audio', 'm_u_source_port','m_d_call_completed', 'm_d_destination_port',
                             'm_d_long_call_indicator','m_d_middle_gap_indicator', 'm_d_short_call_indicator','m_d_one_way_audio', 'm_d_source_port'
                            , 'm_error_code', 'call_indicator', 'start_cell_id', 'last_ho_source_cell_id', 'last_ho_target_cell_id'
                            ,'start_timing_advance_cell', 'trace_id', 'rf_indicator', 'call_indicator_lsr', 'c_failed_lsr']

for col in object_originally_numeric:
    if col in df_merged.columns:
        df_merged[col] = df_merged[col].fillna('UNK')
        df_merged[col] = df_merged[col].astype('object')
    
# Step 8 Finally Print Data type of all columns 
df_object_dtype = pd.DataFrame(df_merged.dtypes, columns=['object_type'])
df_object_unique = pd.DataFrame(df_merged.nunique(), columns=['nunique'])
df_object_dtype = pd.concat([df_object_dtype, df_object_unique], axis =1)
print df_object_dtype

#Step 9 Convert Time Durations to miliseconds or seconds

def convert_to_sec(x):
    try:
        sec = float(x.split(":")[0])*60+float(x.split(":")[1])
        return sec
    except:
        return np.nan

def convert_to_millisec(x):
    try:
        sec = 1000*(float(x.split(":")[0])*60+float(x.split(":")[1]))
        return sec
    except:
        return np.nan
# Step 10 Now, finally plot histogram(Kernel Density Estimation Plots) for Numerical Data 
## I have used Seaborn library to plot the data 
def kdeplot(df, target, feature):
    plt.figure(figsize=(9, 4))
    plt.title("KDE for {}".format(feature))
    ax0 = sns.kdeplot(df[df[target] == 0][feature].dropna(), color= 'navy', label= target+': 0')
    ax1 = sns.kdeplot(df[df[target] == 1][feature].dropna(), color= 'orange', label= target+': 1')
    

print 'Plots for target -> '+str('target = MOS')
for col in df_merged.select_dtypes(['float64', 'int32','int64']):
    print col
    print 'Plots for target -> '+str('target = mos')
    kdeplot(df_merged,'target', col)
    plt.show()
    
    
#Step 11 Plot value count for Categorical Data
def barplot_percentages(df, target, feature, orient='v', axis_name="percentage of records"):
    ratios = pd.DataFrame()
    df=df[df[target].isnull()==False]
    tot_df = len(df)
    keep_vals = df[feature].value_counts().index[:20].tolist()
    df = df[df[feature].isin(keep_vals)]
    g = df.groupby(feature)[target].value_counts().to_frame()
    g = g.rename({target: axis_name}, axis=1).reset_index()
    g[axis_name] = g[axis_name]/tot_df

    f, ax = plt.subplots(figsize=(8, 6))
    plt.tick_params(axis='both', which='major', labelsize=10)
    if orient == 'v':
        ax = sns.barplot(x=feature, y= axis_name, hue=target, data=g, orient=orient)
        for item in ax.get_xticklabels():
            item.set_rotation(90)
        for p in ax.patches:
             ax.annotate(100.0*float("%.4f" % p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=11, color='black', rotation=90, xytext=(0, 20),
                 textcoords='offset points') 

        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
    else:
        ax = sns.barplot(x= axis_name, y=feature, hue=target, data=g, orient=orient)
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])
    ax.plot()
#tgt = target_cols[0]
print 'Plots for target -> '+str('target')
for col in df_merged.select_dtypes('object'):
    print col
    barplot_percentages(df_merged,'target', col)
    plt.show()
    

    
