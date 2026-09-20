
import pandas as pd
import re


#Add a column to variable metadata that just shows values in the values_key
def extract_ints(val):
    return [int(num) for num in re.findall(r'\d+', str(val))]


"""
Domain-knowledge-based imputation for the TBU PUD dataset.

Import into a notebook with:
    from impute_domain import impute_domain_knowledge
"""

import pandas as pd


def impute_domain_knowledge(df):
    """
    Impute missing values using clinical domain knowledge about howrelated variables in the dataset based on my assumptions (listed in report)

    Parameters
    ----------
    df : pandas.DataFrame
        The raw data with missing values to impute. This dataframe should be a copy of the raw dataset

    Returns
    -------
    pandas.DataFrame
        A new dataframe with the domain-knowledge imputations applied
    """

    # AMS: if GCS == 15, AMS = 0. If any AMS indicator is observed (=1), AMS = 1.
    ams_mask = (df['GCSTotal'] == 15) & (df['AMS'].isna())
    df.loc[ams_mask, 'AMS'] = 0
    impute_ams_cols = ['AMSAgitated', 'AMSSleep', 'AMSSlow', 'AMSRepeat', 'AMSOth']
    ams_mask = (df[impute_ams_cols] == 1).any(axis=1) & (df['AMS'].isna())
    df.loc[ams_mask, 'AMS'] = 1

    # Hematoma: if location/size marked "not applicable" (92), assume no hematoma.
    impute_hema_cols = ['HemaLoc', 'HemaSize']
    hema_mask = (df[impute_hema_cols] == 92).any(axis=1) & (df['Hema'].isna())
    df.loc[hema_mask, 'Hema'] = 0

    # Injury severity: pedestrian hit by car (InjuryMech == 2) assumed high severity (3).
    inj_sev_mask = (df['InjuryMech'] == 2) & (df['High_impact_InjSev'].isna())
    df.loc[inj_sev_mask, 'High_impact_InjSev'] = 3

    # History of LOC: 92 -> no LOC (0), 1-4 -> LOC (1), remaining blanks -> no LOC (0).
    loc_mask = (df['LocLen'] == 92) & (df['LOCSeparate'].isna())
    df.loc[loc_mask, 'LOCSeparate'] = 0
    loc_mask = (df['LocLen'] >= 1) & (df['LocLen'] <= 4) & (df['LOCSeparate'].isna())
    df.loc[loc_mask, 'LOCSeparate'] = 1
    df['LOCSeparate'] = df['LOCSeparate'].fillna(0)
    # Duration of LOC: 0 if no history of LOC.
    loc_mask = (df['LOCSeparate'] == 0) & (df['LocLen'].isna())
    df.loc[loc_mask, 'LocLen'] = 0

    # Basilar skull fracture: any indicator = 1 -> SFxBas = 1; all indicators = 92 -> SFxBas = 0.
    impute_sfxbas_cols = ['SFxBasHem', 'SFxBasOto', 'SFxBasPer', 'SFxBasRet', 'SFxBasRhi']
    sfxbas_mask = (df[impute_sfxbas_cols] == 1).any(axis=1) & (df['SFxBas'].isna())
    df.loc[sfxbas_mask, 'SFxBas'] = 1
    sfxbas_mask = (df[impute_sfxbas_cols] == 92).any(axis=1) & (df['SFxBas'].isna())
    df.loc[sfxbas_mask, 'SFxBas'] = 0

    # Palpable skull fracture: 92 (not applicable) -> assume no fracture (0).
    sfxpalp_mask = (df['SFxPalpDepress'] == 92) & (df['SFxPalp'].isna())
    df.loc[sfxpalp_mask, 'SFxPalp'] = 0

    # Headache: severity 92 -> no headache (0); no headache (0) -> severity 92.
    ha_mask = (df['HASeverity'] == 92) & (df['HA_verb'].isna())
    df.loc[ha_mask, 'HA_verb'] = 0
    ha_mask = (df['HA_verb'] == 0) & (df['HASeverity'].isna())
    df.loc[ha_mask, 'HASeverity'] = 92

    # Acting normal: predictor for children under 2. AMS = 1 -> ActNorm = 0.
    act_norm_mask = (df['AgeTwoPlus'] == 1) & (df['AMS'] == 1) & (df['ActNorm'].isna())
    df.loc[act_norm_mask, 'ActNorm'] = 0

    # Vomit: any non-applicable (92) observation -> assume no vomiting (0).
    impute_vomit_cols = ['VomitNbr', 'VomitStart', 'VomitLast']
    vomit_mask = (df[impute_vomit_cols] == 92).any(axis=1) & (df['Vomit'].isna())
    df.loc[vomit_mask, 'Vomit'] = 0

    return df


def impute_from_median(df, columns, variable_metadata):
    """
    Fill missing values in the given columns with each column's median
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to impute.
    columns : list of str
        Columns to impute.
    variable_metadata : pandas.DataFrame
        Metadata dataframe for variables
 
    Returns
    -------
    pandas.DataFrame
        A new dataframe with missing values in `columns` filled with
        their median.
    """
    df = df.copy()
    for variable in columns:
        med_value = variable_metadata['50%'][variable]
        df[variable] = df[variable].fillna(med_value)
    return df
 
 
def impute_from_distribution(df, columns, random_state=None):
    """
    Fill missing values in the given columns by randomly sampling from
    each column's own observed (non-missing) values.
 
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to impute.
    columns : list of str
        Columns to impute.
    random_state : int, optional
        Seed for reproducible sampling.
 
    Returns
    -------
    pandas.DataFrame
        A new dataframe with missing values in `columns` filled by
        sampling from their observed distribution.
    """
    df = df.copy()
    for variable in columns:
        observed = df[variable].dropna()
        n_missing = df[variable].isnull().sum()
        sampled_values = observed.sample(n=n_missing, replace=True, random_state=random_state)
        df.loc[df[variable].isnull(), variable] = sampled_values.values
    return df
 
 
def drop_missing_rows(df, columns):
    """
    Drop rows that have a missing value in any of the given columns.
 
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to filter.
    columns : list of str
        Columns to check for missing values.
 
    Returns
    -------
    pandas.DataFrame
        A new dataframe with rows dropped where any of `columns` is missing.
    """
    return df.dropna(subset=columns)
 