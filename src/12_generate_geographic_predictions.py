"""FireWatch AI: export 2024 held-out and Jan 2025 experimental grid scores.

Run from the repository root:
    python src/12_generate_geographic_predictions.py

The saved Apache Spark MLlib model is loaded (never retrained); only small
precomputed grid-month tables are processed. No Spark parquet writes are used.
"""

import os
import sys
from pathlib import Path

# Spark on Windows must use the active virtual environment's Python worker.
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
os.environ['SPARK_LOCAL_HOSTNAME'] = 'localhost'

if Path(r'C:\hadoop\bin\winutils.exe').is_file():
    os.environ.setdefault('HADOOP_HOME', r'C:\hadoop')
    hadoop_bin = str(Path(r'C:\hadoop\bin'))
    if hadoop_bin.lower() not in os.environ.get('PATH', '').lower().split(os.pathsep):
        os.environ['PATH'] = hadoop_bin + os.pathsep + os.environ.get('PATH', '')

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'data' / 'processed' / 'prediction_dataset.parquet'
MODEL_DIR = ROOT / 'models' / 'geographic_rf'
OUTPUT_TEST = ROOT / 'data' / 'processed' / 'geographic_predictions_2024_test.parquet'
OUTPUT_JAN = ROOT / 'data' / 'processed' / 'geographic_predictions_2025_01.parquet'
EXPECTED_CELLS = 822
FEATURES = ['lat_grid', 'lon_grid', 'month', 'lag_1', 'lag_2', 'lag_3']

SCHEMA = StructType([
    StructField('lat_grid', IntegerType(), False),
    StructField('lon_grid', IntegerType(), False),
    StructField('month', IntegerType(), False),
    StructField('lag_1', DoubleType(), False),
    StructField('lag_2', DoubleType(), False),
    StructField('lag_3', DoubleType(), False),
    StructField('actual_detection', IntegerType(), True),
    StructField('actual_detections', IntegerType(), True),
])


def validate_data(frame):
    required = {'lat_grid', 'lon_grid', 'month', 'lag_1', 'lag_2', 'lag_3', 'detections', 'target'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'Prediction dataset is missing columns: {missing}')
    if frame[list(required)].isna().any().any():
        raise ValueError('Prediction dataset has missing required values.')
    if frame.duplicated(['lat_grid', 'lon_grid', 'month']).any():
        raise ValueError('Duplicated (grid, month) rows in prediction dataset.')
    if set(frame['month'].astype(int).unique()) != set(range(4, 13)):
        raise ValueError('Expected prediction dataset for April–December 2024.')
    counts = frame.groupby('month').size()
    if len(counts) != 9 or not counts.eq(EXPECTED_CELLS).all():
        raise ValueError(f'Expected {EXPECTED_CELLS} grid cells for each month, got {counts.to_dict()}.')
    if not frame['target'].isin([0, 1]).all():
        raise ValueError('Target labels must be 0 or 1.')
    if not (frame['target'].astype(int) == (frame['detections'] > 0).astype(int)).all():
        raise ValueError('Target is inconsistent with detection counts.')


def make_january_rows(history):
    # Historical observations must be exactly October, November, December 2024.
    recent = history.loc[history['month'].isin([10, 11, 12]), [
        'lat_grid', 'lon_grid', 'month', 'detections'
    ]]
    pivot = recent.pivot(index=['lat_grid', 'lon_grid'], columns='month', values='detections')
    if list(pivot.columns) != [10, 11, 12] or len(pivot) != EXPECTED_CELLS:
        raise ValueError('Incomplete October–December history for January scenario.')
    if pivot.isna().any().any():
        raise ValueError('Some January scenario historical months are missing.')
    january = pivot.reset_index()[['lat_grid', 'lon_grid']].copy()
    january['month'] = 1
    january['lag_1'] = pivot[12].to_numpy(dtype=float)
    january['lag_2'] = pivot[11].to_numpy(dtype=float)
    january['lag_3'] = pivot[10].to_numpy(dtype=float)
    january['actual_detection'] = None  # Not known / not validated by this project.
    january['actual_detections'] = None
    return january


def score(spark, model, frame, year):
    rows = []
    for r in frame.itertuples(index=False):
        actual = None if pd.isna(r.actual_detection) else int(r.actual_detection)
        count = None if pd.isna(r.actual_detections) else int(r.actual_detections)
        rows.append((
            int(r.lat_grid), int(r.lon_grid), int(r.month),
            float(r.lag_1), float(r.lag_2), float(r.lag_3), actual, count
        ))
    raw = spark.createDataFrame(rows, schema=SCHEMA)
    assembled = VectorAssembler(inputCols=FEATURES, outputCol='features').transform(raw)
    predictions = model.transform(assembled)
    output = predictions.select(
        'lat_grid', 'lon_grid', 'month', 'lag_1', 'lag_2', 'lag_3',
        'actual_detection', 'actual_detections',
        F.col('prediction').cast('int').alias('prediction'),
        vector_to_array('probability')[0].alias('probability_no_detection'),
        vector_to_array('probability')[1].alias('probability_detection'),
    ).toPandas()
    output['forecast_year'] = year
    output = output.rename(columns={
        'lag_1': 'previous_1_month_detections',
        'lag_2': 'previous_2_month_detections',
        'lag_3': 'previous_3_month_detections',
    })
    output = output.sort_values(['month', 'lat_grid', 'lon_grid']).reset_index(drop=True)
    if len(output) != len(frame) or output[['lat_grid', 'lon_grid', 'month']].duplicated().any():
        raise ValueError('Model scoring produced an unexpected number of grid-month rows.')
    if not output['probability_detection'].between(0, 1).all():
        raise ValueError('Model scoring produced invalid scores.')
    if not (output['probability_detection'] + output['probability_no_detection']).between(.99999, 1.00001).all():
        raise ValueError('Model probability vectors do not sum to one.')
    return output


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f'Missing file: {INPUT}')
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f'Missing saved Spark model: {MODEL_DIR}')
    source = pd.read_parquet(INPUT)
    validate_data(source)
    print('\n=== PREDICTION DATASET ===')
    print('Rows:', len(source), '| geographic cells:', EXPECTED_CELLS)

    test = source.loc[source['month'].isin([11, 12])].copy()
    test['actual_detection'] = test['target']
    test['actual_detections'] = test['detections']
    january = make_january_rows(source)

    spark = (SparkSession.builder.master('local[2]')
             .appName('FireWatch-Export-Geographic-Predictions')
             .config('spark.sql.shuffle.partitions', '4')
             .getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    try:
        model = RandomForestClassificationModel.load(str(MODEL_DIR))
        if model.numFeatures != len(FEATURES):
            raise ValueError(f'Saved model expects {model.numFeatures} features, not {len(FEATURES)}.')
        print('\n=== SCORING NOV-DEC 2024 HOLDOUT ===')
        test_scores = score(spark, model, test, year=2024)
        print('Scored rows:', len(test_scores))
        observed = test_scores['actual_detection'].astype(int)
        accuracy = (test_scores['prediction'].astype(int) == observed).mean()
        persistence = (test_scores['previous_1_month_detections'] > 0).astype(int)
        baseline_accuracy = (persistence == observed).mean()
        print(f'Model accuracy: {accuracy:.4f}')
        print(f'Persistence baseline accuracy: {baseline_accuracy:.4f}')
        if not .80 < accuracy < .83:
            raise ValueError('Reloaded model test accuracy differs from previously verified ~0.8139. Check feature order / model version.')

        print('\n=== SCORING JANUARY 2025 SCENARIO ===')
        jan_scores = score(spark, model, january, year=2025)
        print('Scored rows:', len(jan_scores))
    finally:
        spark.stop()

    # Write outputs only after BOTH scoring operations have passed validation.
    OUTPUT_TEST.parent.mkdir(parents=True, exist_ok=True)
    test_scores.to_parquet(OUTPUT_TEST, index=False)
    jan_scores.to_parquet(OUTPUT_JAN, index=False)
    print('\nSaved:', OUTPUT_TEST)
    print('Saved:', OUTPUT_JAN)
    print('WARNING: January 2025 is not externally validated; no real-world probability claims.')
    print('\n=== COMPLETE ===')


if __name__ == '__main__':
    main()
