import pyspark
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.mllib.evaluation import BinaryClassificationMetrics


def df_with_proba_column(model, df, probabilities_col, negative_downsampling_rate):
    def get_recalibrated_probability(p):
        q = p / (p + (1-p)/negative_downsampling_rate)
        return float(q)

    df = model.transform(df)
    df = df.withColumn(
        'proba', 
        F.udf(lambda v: get_recalibrated_probability(float(v[1])), FloatType())(F.col(probabilities_col)))
    return df


def rocauc(model, df, probabilities_col='probability', negative_downsampling_rate=1.0):
    df = df_with_proba_column(model, df, probabilities_col, negative_downsampling_rate)

    preds_and_labels = df\
        .select(F.col('proba'), F.col('label').cast('float')) \
        .rdd.map(lambda row: (row[0], row[1]))
    
    metrics = BinaryClassificationMetrics(preds_and_labels)

    return metrics.areaUnderROC


def logloss(model, df, probabilities_col='probability', negative_downsampling_rate=1.0):
    df = df_with_proba_column(model, df, probabilities_col, negative_downsampling_rate)

    df = df.withColumn('logloss',
                       - F.col('label') * F.log(F.col('proba')) - (1. - F.col('label')) * F.log(1. - F.col('proba')))

    return df.agg(F.mean('logloss')).first()[0]


def ne(model, df, probabilities_col='probability', negative_downsampling_rate=1.0):
    ll = logloss(model, df, probabilities_col, negative_downsampling_ratio)
    p = df.select(F.mean('label')).first()[0]
    ll_baseline =  -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return ll / ll_baseline


def calibration(model, df, probabilities_col='probability', negative_downsampling_rate=1.0):
    empirical_ctr = df.agg(F.mean('label')).first()[0]
    if empirical_ctr == 0.0:
        raise ValueError("The calibration is undefined when empirical_ctr is 0.")

    df = df_with_proba_column(model, df, probabilities_col, negative_downsampling_rate)
    df = df.withColumn('proba_prediction', 
                       F.udf(lambda v: 1.0 if v >= 0.5 else 0.0, FloatType())(F.col('proba')))
    estimated_ctr = df.agg(F.mean('proba_prediction')).first()[0]
    return abs(estimated_ctr - empirical_ctr) / empirical_ctr
