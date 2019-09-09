# reading the input file and separating the train and test data.
import pandas as pd
df = pd.read_csv('Data.csv', delimiter=';')
nan_df = df[df['hits'] == '\\N']
df = df[df['hits'] != '\\N']


def imp(counter):
    '''
    A function that returns a importance weighted value (based on frequency) of each element in a list.
    :param counter: collections.Counter object
    :return: a dict importance weighted values
    '''
    imp_dict = {}
    for key, value in dict(counter).items():
        imp_dict[key] = value / sum(counter.values())
    return imp_dict

# Switfter speeds up the .apply() in pandas.
import swifter
def process(df):

    '''
    :param df: Any of the train or test dataframes
    :return: Processed dataframe either for training or prediction
    '''
    df['session_durantion'] = df['session_durantion'].replace(to_replace='\\N',value =0)

    df['path_id_set'] = df['path_id_set'].str.split(';')
    df['path_id_set'].loc[df['path_id_set'].isnull()] = df['path_id_set'].loc[df['path_id_set'].isnull()].apply(lambda x: [])
    df['path_length'] = df['path_id_set'].swifter.set_dask_scheduler('processes').set_npartitions(32).allow_dask_on_strings(enable=True).apply(len)


    import itertools
    path_ids=[]
    df['path_id_set'].apply(lambda x : path_ids.append(x))
    path_ids_list =  list(itertools.chain(*path_ids))

    import collections
    counter = collections.Counter(path_ids_list)


    imp_dict = imp(counter)

    df['path_imp'] = df['path_id_set'].swifter.set_dask_scheduler('processes').set_npartitions(32).allow_dask_on_strings(enable=True).apply(lambda x : sum([imp_dict[i] for i in x]))


    entry_page_counter = collections.Counter(df['entry_page'].tolist())
    entry_page_dict = imp(entry_page_counter)
    df['entry_page_imp'] = df['entry_page'].swifter.set_dask_scheduler('processes').set_npartitions(32).allow_dask_on_strings(enable=True).apply(lambda x : entry_page_dict[x])

    df.drop(columns = ['entry_page','path_id_set'],inplace = True )
    df = pd.get_dummies(df, prefix = ['locale','day_of_week','agent_id','traffic_type'], columns = ['locale','day_of_week','agent_id','traffic_type'] )


    # An attempt to use log form of the feature 'session_durantion', as it has better correlation with 'hits'
    # Models performed better with 'session_durantion' rather than 'log_session_durantion', so switching back.

    #df['session_durantion'] = pd.to_numeric(df['session_durantion'])
    #df['hits'] = pd.to_numeric(df['hits'])
    # import numpy as np
    # df['log_session_durantion'] =df['session_durantion'].apply(np.log)
    # df['log_session_durantion'] = df['log_session_durantion'].replace(-np.inf,0)

    df = df [['row_num', 'hour_of_day','session_durantion','path_length',
           'path_imp', 'entry_page_imp', 'locale_L1', 'locale_L2', 'locale_L3',
           'locale_L4', 'locale_L5', 'locale_L6','day_of_week_Monday',
            'day_of_week_Tuesday','day_of_week_Wednesday','day_of_week_Thursday',
            'day_of_week_Friday','day_of_week_Saturday', 'day_of_week_Sunday',
           'agent_id_0', 'agent_id_1', 'agent_id_2', 'agent_id_3', 'agent_id_4',
           'agent_id_5', 'agent_id_7', 'agent_id_8', 'agent_id_9', 'agent_id_10',
           'agent_id_11', 'agent_id_12', 'agent_id_13', 'agent_id_14',
           'agent_id_15', 'traffic_type_1', 'traffic_type_2', 'traffic_type_3',
           'traffic_type_4', 'traffic_type_6', 'traffic_type_7',
           'traffic_type_10','hits']]

    return df

#df.corr().iloc[:,-1:]
process(df).to_csv('train.csv',index = False)
process(nan_df).to_csv('test.csv',index = False)
#print(df)