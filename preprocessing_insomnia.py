import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

import random

CAT_COLS = [
    # Accultiration
    "ACD010A",
    "ACD010B",
    "ACD010C",
    "ACD040",


    # Alcohol
    "ALQ111",
    "ALQ151",

    # Blood pressure & cholosterol
    "BPQ020",
    "BPQ030",
    "BPQ150",
    "BPQ080",
    "BPQ101D",

    # Current health status
    "HSQ590",

    # Diabetes
    "DIQ010",
    "DIQ160",
    "DIQ180",
    "DIQ050",
    "DIQ070",

    # Diet behavior & nutrition

    # Go over this again, some of them are appareantly ordinal
    "DBQ010",
    "DBQ073A",
    "DBQ073B",
    "DBQ073C",
    "DBQ073D",
    "DBQ073E",
    "DBQ073U",
    "DBQ301",
    "DBQ330",
    "DBQ360",
    "DBQ370",
    "DBQ400",
    "DBQ424",
    "DBQ930",
    "DBQ935",
    "DBQ940",
    "DBQ945",

    # Food security
    "FSD041",
    "FSD061",
    "FSD071",
    "FSD081",
    "FSD092",
    "FSD151",
    "FSQ165",
    "FSQ012",
    "FSD230",
    "FSD162",
    "FSQ760",
    "FSQ653",
    "FSD660ZC",
    "FSD675",
    "FSD680",
    "FSQ690",

    # Functioning
    "FNQ050",
    "FNDADI",
    "FNDAEDI",
    "FNDCDI",

    # Health insurance
    "HIQ011",
    "HIQ210",

    # Hospital utilization & access to care
    "HUQ030",
    "HUQ042",
    "HUQ055",
    "HUQ090",

    # Income
    "INQ300",

    # Kidney conditions - urology
    "KIQ022",
    "KIQ025",
    "KIQ042",
    "KIQ044",




    # Medical conditions
    "MCQ010",
    "MCQ035",
    "MCQ040",
    "MCQ050",
    "AGQ030",
    "MCQ053",
    "MCQ149",
    "MCQ160A",
    "MCQ195",
    "MCQ160B",
    "MCQ160C",
    "MCQ160D",
    "MCQ160E",
    "MCQ160F",
    "MCQ160M",
    "MCQ170M",
    "MCQ160P",
    "MCQ160L",
    "MCQ170L",
    "MCQ500",
    "MCQ510A",
    "MCQ510B",
    "MCQ510C",
    "MCQ510D",
    "MCQ510E",
    "MCQ510F",
    "MCQ550",
    "MCQ560",
    "MCQ220",
    "MCQ230A",
    "MCQ230B",
    "MCQ230C",
    "MCQ230D",
    "OSQ230",

    # Occupation
    "OCD150",
    "OCQ210",
    "OCQ383",

    # Prescription medications
    "RXQ033",

    # Preventive aspirin use
    "RXQ510",
    "RXQ515",
    "RXQ520",

    # Reproductive health
    "RHQ031",
    "RHD043",
    "RHQ078",
    "RHQ131",
    "RHD143",
    "RHQ200",
    "RHD280",
    "RHQ305",

    # Smoking - cigarette use
    "SMQ020",
    "SMD100MN",

    # Smoking - recent tobacco use
    "SMQ681",
    "SMQ690A",
    "SMQ690B",
    "SMQ690C",
    "SMQ690G",
    "SMQ846",
    "SMQ851",
    "SMQ690D",
    "SMQ690E",
    "SMQ690K",
    "SMQ863",
    "SMQ690F",
    "SMDANY",

    # Weight history
    "WHQ070"
]

# you might want to reverse the scales for some of them
# change it so that never becomes the smallest number
# 5=poor should stay as is => higher = worse health
# higher the value more risk that person has for insomnia
ORDINAL_COLS = [
    # Alcohol
    "ALQ121",
    "ALQ142",
    "ALQ270",
    "ALQ280",

    # Diet behaviour & nutrition
    "DBQ390",
    "DBQ421",

    # Food security
    "FSD032A",
    "FSD032B",
    "FSD032C",
    "FSD052",
    "FSD102",
    "FSDAD",

    # Functioning
    # 1 = no difficulty, 4 = too difficult
    "FNQ021",
    "FNQ041",
    "FNQ060",
    "FNQ080",
    "FNQ160",
    "FNQ100",
    "FNQ110",
    "FNQ120",
    "FNQ170",
    "FNQ180",
    "FNQ190",
    "FNQ130",
    "FNQ200",
    # how often anxious, worried?
    "FNQ140",
    "FNQ150",
    "FNQ410",
    "FNQ430",
    "FNQ440",
    "FNQ450",
    "FNQ460",
    "FNQ470",
    "FNQ480",
    "FNQ490",
    # how often worried? 1 = daily, 4 = few times a year
    # fix the ordering
    "FNQ510",
    # fix the ordering
    "FNQ520",
    # fix
    "FNQ530",
    # fix
    "FNQ540",


    # Hospital utilization & access to care
    "HUQ010",

    # Income
    "INDFMMPC",
    "IND310",

    # Kidney conditions - urology
    # how often 1 = never
    "KIQ005",
    # how much daily activites affected 1 = not at all
    "KIQ052",

    # Mental Health - Depression Screener
    "DPQ010",
    "DPQ020",
    "DPQ030",
    "DPQ040",
    "DPQ050",
    "DPQ060",
    "DPQ070",
    "DPQ080",
    "DPQ090",
    "DPQ100",

    # Oral health
    # Rate the health of your teeth and gums
    # excellent => 1, poor => 5
    "OHQ845",
    # same answer type
    # How often last yr. had aching in mouth?
    # very often difficulty => 1, never => 5
    "OHQ620",
    "OHQ630",
    "OHQ640",
    "OHQ660",
    "OHQ670",
    "OHQ680",

    # Smoking - cigarette use
    "SMQ040",
    "SMQ621",

    # Smoking - recent tobacco use
    "SMQ725"

]

NUM_COLS = [
    # Alcohol
    "ALQ130",
    "ALQ170",

    # Diabetes
    "DID040",
    "DID060",

    # Diet behaviour & nutrition
    "DBD030",
    "DBD041",
    "DBD050",
    "DBD055",
    "DBD061",
    "DBD381",
    "DBD411",

    # Food security
    "FSD165N",
    "FSD012N",
    "FSD795",
    "FSD225",
    "FSD230N",
    "FSD235",
    "FSD760N",
    "FSD670ZC",
    "FSQ695",

    # Housing characteristics
    "HOD051",

    # Income
    "INDFMMPI",

    # Kidney conditions - urology
    "KIQ481",

    # Occupation
    "OCQ180",
    "OCQ215",

    # Physical Activity
    "PAD790Q",
    "PAD800",
    "PAD810Q",
    "PAD820",
    "PAD680",

    # Physical Activity Youth
    "PAQ706",
    "PAQ711",

    # Prescription medications
    "RXQ050",

    # Reproductive health
    "RHQ010",
    "RHQ060",
    "RHD167",
    "RHQ332",

    # Smoking - cigarette use
    "SMD650",
    "SMD630",

    # Smoking - recent tobacco use
    "SMQ710",
    "SMQ720",
    "SMQ740",
    "SMQ770",
    "SMQ845",
    "SMQ849",

    # Weight history
    "WHD010",
    "WHD020",
    "WHD050"

]

unit_of_measure = [
    # diabetes
    "DIQ060U",

    # physical activity
    "PAD790U",
    "PAD810U",
]


# # drop the columns that contain NaN values above a specified threshold
def drop_cols_na(data, threshold):
    cols = [col for col in data.columns if data[col].isna().sum()/data[col].shape[0]>threshold]
    data = data.drop(cols, axis=1)
    return data

def sep_target(data, target_cols):
    X = data.drop(columns=target_cols)
    y = data[target_cols]
    return X, y

# cleaning X data
def clean_data(data, threshold):

    data = data.copy()

    # converting bytestrings
    obj_cols = data.select_dtypes(include=["object"]).columns.to_list()
    for col in obj_cols:
        data[col] = data[col].apply(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)
        data[col] = data[col].replace(['', ' '], np.nan)


    # valid 7 columns: columns which 7 corresponds to actual numerical answer
    VALID_SEVEN_COLS = [
        # Alcohol
        "ALQ121",
        "ALQ130",
        "ALQ142",
        "ALQ270",
        "ALQ280",
        "ALQ170",
        # Diabetes
        "DID040",
        # Food security
        "FSD165N",
        "FSD012N",
        "FSD230N",
        "FSD795",
        "FSQ695",
        # Housing characteristics
        "HOD051",
        # Occupation
        "OCQ215",
        "OCQ383",
        # Physical activity youth
        "PAQ706",
        "PAQ711",
        # Reproductive health
        "RHQ010",
        "RHD043",
        # Smoking - cigarette use
        "SMD641",
        "SMQ621",
        # Smoking - recent tobacco use
        "SMQ720",
        ]

    # valid 9 columns: columns which 9 corresponds to actual numerical answer
    VALID_NINE_COLS = [
        # Alcohol
        "ALQ121",
        "ALQ130",
        "ALQ142",
        "ALQ270",
        "ALQ280",
        "ALQ170",
        # Diabetes
        "DID040",
        # Food security
        "FSD795",
        "FSQ695",
        # Housing characteristics
        "HOD051",
        # Physical activity youth
        "PAQ711",
        # Reproductive health
        "RHQ010",
        "RHD043",
        # Smoking - cigarette use
        "SMD641",
        "SMD630",
    ]

    # replace 777777, 999999 with nans
    # numbers bigger than 77, 99
    VERY_BIG_VALID_NUMBER_COLS = [
        # Diet behaviour & nutrition
        "DBD030",
        "DBD041",
        "DBD050",
        "DBD055",
        "DBD061",
        # Food security
        "FSD795",
        "FSD225",
        "FSD235",
        "FSD670ZC",
        # Occupation
        "OCQ180",
        # Physical Activity
        "PAD790Q",
        "PAD800",
        "PAD810Q",
        "PAD820",
        "PAD680",
        # Smoking - cigarette use
        "SMD650",
        # Weight history
        "WHD010",
        "WHD020",
        "WHD050"
    ]

    general_missing_answer_numbers = [7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999, 777777, 999999]
    valid_seven_nine = [77, 99, 777, 999, 7777, 9999, 77777, 99999, 777777, 999999]
    very_big_valid_numbers = [777, 999, 7777, 9999, 77777, 99999, 777777, 999999]


    for col in data.columns:
        if col in (set(VALID_SEVEN_COLS) | set(VALID_NINE_COLS)):
            data[col] = data[col].replace(valid_seven_nine, np.nan)
        elif col in set(VERY_BIG_VALID_NUMBER_COLS):
            data[col] = data[col].replace(very_big_valid_numbers, np.nan)
        else:
            data[col] = data[col].replace(general_missing_answer_numbers, np.nan)


    # finding out the approximate minute of physical activity because there is no numerical data for them.
    convert_frequency = {
    "D": 7,
    "W": 1,
    "M": 1/4,
    "Y": 1/52
    }

    # if one of the answers is NaN in the end it ends up being NaN
    # physicality = frequency * duration
    # if one of these are missing physicality cannot be caclculated properly so dropping columns shouldn't create a problem
    data["approximate_freq_moderate_LTPA"] = data["PAD790Q"] * data["PAD790U"].map(convert_frequency)
    data["approximate_freq_vigorous_LTPA"] = data["PAD810Q"] * data["PAD810U"].map(convert_frequency)

    data["approximate_mins_moderate_LTPA"] = data["approximate_freq_moderate_LTPA"] * data["PAD800"]
    data["approximate_mins_vigorous_LTPA"] = data["approximate_freq_vigorous_LTPA"] * data["PAD820"]

    data = data.drop(columns= ["approximate_freq_moderate_LTPA", "PAD790Q", "PAD790U","PAD800",
                        "approximate_freq_vigorous_LTPA", "PAD810Q", "PAD810U", "PAD820"])


    # finding out the the number of months/years the participant has been taking insulin
    convert_month_year = {
        # month
        1: 1/12,
        # year
        2: 1
    }

    data["insulin_time"] = data["DID060"] * data["DIQ060U"].map(convert_month_year)
    # handle the answers that said less than a month
    data.loc[data["DID060"] == 666, "insulin_time"] = 0
    data = data.drop(columns=["DID060", "DIQ060U"])

    edited_numerical = ["approximate_mins_moderate_LTPA", "approximate_mins_vigorous_LTPA", "insulin_time"]



    # diet behaviour & nutrition => still breastfeeding
    # the question is asking the age stopped breastfeeding
    data.loc[data["DBD030"] == 0, "DBD030"] = np.nan


    # fixing the ordering of functioning
    order_corrected = {
        # a little
        1:1,
        # somewhere between a little and a lot
        3:2,
        # a lot
        2:3
    }

    data["FNQ520_fixed"] = data["FNQ520"].map(order_corrected)
    data["FNQ540_fixed"] = data["FNQ540"].map(order_corrected)
    data = data.drop(columns=["FNQ520", "FNQ540"], errors="ignore")

    edited_ordinal = ["FNQ520_fixed", "FNQ540_fixed"]



    # drop unnecessary columns
    # also check if this column is dropped => DIQ159, DIQ065, DBD265A, DBD355, DBQ422, DBD710,
    # FSQBOX1, FSQBOX2, FSQBOX5, FSQBOX6, HUQ085, MCQ145, MCQ157, MCQ515, OCQ200, SMAQUEX2
    columns_to_drop = ["SEQN", "DIQ159", "DIQ065", "DBD265A", "DBD355", "DBQ422", "DBD710", "FSQBOX1", "FSQBOX2",
                       "FSQBOX5", "FSQBOX6", "HUQ085", "MCQ145", "MCQ157", "MCQ515", "OCQ200", "SMAQUEX2", "KIQ010", "KIQ048A"]
    unit_of_measure = [
    # diabetes
    "DIQ060U"
    ]


    data = data.drop(columns=columns_to_drop, errors="ignore")
    data = data.drop(columns=unit_of_measure, errors="ignore")


    # remove the columns that have NaN values above a certain threshold
    data = drop_cols_na(data, threshold)

    data = data.loc[:,~data.columns.duplicated()]


    # DO NOT FORGET TO ADD THE COLUMNS HERE
    # changing the order of some ordinal columns so that they overall stay consistent
    # higher numer = higher risk
    ordinal_cols_to_reverse = [
        # Alcohol
    "ALQ121",
    "ALQ142",
    "ALQ270",
    "ALQ280",

    # Food security
    "FSD032A",
    "FSD032B",
    "FSD032C",
    "FSD052",
    "FSD102",

    # functioning
    "FNQ140",
    "FNQ150",
    "FNQ510",
    "FNQ530",

    # Income
    "INDFMMPC",
    "IND310",

    # oral health
    "OHQ620",
    "OHQ630",
    "OHQ640",
    "OHQ660",
    "OHQ670",
    "OHQ680",

    # Smoking - cigarette use
    "SMQ040",

    # Smoking - recent tobacco use
    "SMQ725"
    ]
    for col in ordinal_cols_to_reverse:
        if col not in data.columns:
            continue
        unique_vals = data[col].dropna().unique()

        min_val = unique_vals.min()
        max_val = unique_vals.max()

        new_mapping = {
            value: min_val + max_val - value for value in unique_vals
        }
        data[col] = data[col].map(new_mapping)

    # make the final lists

    categorical_cols = [col for col in CAT_COLS if col in data.columns]
    ordinal_cols = [col for col in ORDINAL_COLS if col in data.columns]
    numerical_cols = [col for col in NUM_COLS if col in data.columns]


    ordinal_cols += [col for col in edited_ordinal if col in data.columns]
    numerical_cols += [col for col in edited_numerical if col in data.columns]


    # if there is a leftover, assign it to categorical columns

    known_cols = set(categorical_cols) | set(ordinal_cols) | set(numerical_cols)

    categorical_cols += [col for col in data.columns if col not in known_cols]

    # check overlaps
    print(len(set(categorical_cols) & set(ordinal_cols)) == 0)
    print(len(set(categorical_cols) & set(numerical_cols)) == 0)
    print(len(set(numerical_cols) & set(ordinal_cols)) == 0)

    # check if the length matches correctly
    print(set(categorical_cols) | set(ordinal_cols) | set(numerical_cols) == set(data.columns))


    return data, categorical_cols, ordinal_cols, numerical_cols


# clean our target columns

def clean_targets(data):
    data = data.copy()
    # converting bytestrings
    obj_cols = data.select_dtypes(include=["object"]).columns.to_list()
    for col in obj_cols:
        data[col] = data[col].apply(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)
        data[col] = data[col].replace(['', ' '], np.nan)

    # handle our target columns
    data = data.replace([77777, 99999, " ", "", "."], np.nan)

    TIME_COLS = [
        # wake time
        "SLQ310",
        # sleep time
        "SLQ300",
        # sleep time - weekends
        "SLQ320",
        # wake time - weekends
        "SLQ330"
    ]

    # handling actual string b""s
    for time_col in TIME_COLS:
        data[time_col] = data[time_col].astype("string").str.replace(r'\bb', '', regex= True).str.replace("'", '', regex= True)

    return data


def hours_to_minutes(data, column):
    time = data[column].astype("string").str.split(':', expand=True)
    hours = pd.to_numeric(time[0], errors='coerce')
    minutes = pd.to_numeric(time[1], errors='coerce')

    full_minutes = hours * 60 + minutes

    return full_minutes

def minutes_after_midnight(minutes):
    time_between = minutes % (24*60)

    return time_between

# binary

def label_construction(data):

    # CRITERIA #1:
    # Difficulty initiating sleep.
    # (In children, this may manifest as difficulty
    # initiating sleep without caregiver intervention.)

    # late criteria
    late = (2 * 60 + 30)

    # sleep time - weekday
    t_st = hours_to_minutes(data, "SLQ300")
    late_sleep_weekday = (minutes_after_midnight(t_st) >= late) & (minutes_after_midnight(t_st) < 6*60)


    # sleep time - weekends
    t_st_w = hours_to_minutes(data, "SLQ320")
    late_sleep_weekdend = (minutes_after_midnight(t_st_w) >= late) & (minutes_after_midnight(t_st_w) < 6*60)

    late_sleep_always = (late_sleep_weekday | late_sleep_weekdend)

    # CRITERIA #3:
    # Early-morning awakening with inability to return to sleep.

    # early criteria

    early = (5 * 60 + 30)

    # wake time - weekday
    t_wt = hours_to_minutes(data, "SLQ310")
    wake_early_weekday = minutes_after_midnight(t_wt) <= early


    # wake time - weekends
    t_wt_w = hours_to_minutes(data, "SLQ330")
    wake_early_weekend = minutes_after_midnight(t_wt_w) <= early

    wake_early_always = (wake_early_weekday | wake_early_weekend)


    # EXTRA CRITERIA: COMPARING STAYING IN BED TO SLEEP HOURS
    total_min_spent_in_bed_weekday = (t_wt - t_st) % (24*60)

    sleep_minutes_weekday = data["SLD012"] * 60

    # how much of time in bed this person was not asleep

    awake_in_bed_weekday = sleep_minutes_weekday + 90 <= total_min_spent_in_bed_weekday

    total_min_spent_in_bed_weekend = (t_wt_w - t_st_w) % (24*60)

    sleep_minutes_weekend = data["SLD013"] * 60

    # how much of time in bed this person was not asleep

    awake_in_bed_weekend = sleep_minutes_weekend + 90 <= total_min_spent_in_bed_weekend

    awake_overall = (awake_in_bed_weekday | awake_in_bed_weekend)


    # if it is insomnia
    # i did not want to be too strict about this
    insomnia = (late_sleep_always | wake_early_always | awake_overall)

    insomnia = insomnia.fillna(False).astype(int)

    print(insomnia.value_counts())

    return insomnia


def get_preprocessed_pipeline(model, categorical_columns, numerical_columns, ordinal_columns):
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    scaler = StandardScaler()

    preprocess = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ]), numerical_columns),

        ('ord', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ]), ordinal_columns),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', one_hot_encoder)
        ]), categorical_columns),
    ],
    remainder="drop",
    )

    return Pipeline(steps=[("preprocessing", preprocess), ("training", model)])


pipeline = get_preprocessed_pipeline(model, X_categorical, X_numerical, X_ordinal)
pipeline.fit(X_clean, y)

def get_preprocessed_insomnia_data( threshold = 0.4 ):
    data = pd.read_csv('processed_data/insomnia_data.csv')
    target_cols = ["SLQ300", "SLQ310", "SLD012", "SLQ320", "SLQ330", "SLD013"]

    # separate target columns
    X_unclean, y_targets = sep_target(data, target_cols)

    X_clean, X_categorical, X_ordinal, X_numerical = clean_data(X_unclean, threshold)
    y_targets = clean_targets(y_targets)
    y = label_construction(y_targets)

    # there is not going to noise since we generate binary labels
    # explain the inherent shortcomings

    # check if the length matches
    len(X_categorical) + len(X_ordinal) + len(X_numerical) == X_clean.shape[1]