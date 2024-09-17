import momepy
import libpysal
import pandas as pd

def merge_all_metrics(tessellations, buildings, streets, junctions):
    merged = buildings.merge(
        tessellations.drop(columns=['geometry', ]), 
        left_index=True, right_on='tID', how='left'
        )
    merged = merged.merge(
    streets.drop(columns=['geometry','node_start', 'node_end']),
    left_on='street_index', right_index=True, how='left'
    )
    merged = merged.merge(
        junctions.drop(columns=['x','y','geometry','nodeID','cluster']),
        left_on='junction_index', right_index=True, how='left'
    )
    return merged

def compute_percentiles(merged, queen_3):
    percentiles = []
    for column in merged.columns.drop(
        [
            'geometry',
            'street_index',
            'junction_index',
            'mm_len',
        ]
    ):
        print(column)
        perc = momepy.percentile(merged[column], queen_3)
        perc.columns = [f"{column}_" + str(x) for x in perc.columns]
        percentiles.append(perc)

    percentiles_joined = pd.concat(percentiles, axis=1)
    metrics_df = pd.concat([merged.drop(
        columns=['geometry', 'street_index', 'junction_index', 'mm_len',]), 
        percentiles_joined], axis=1)
    return metrics_df

def standardize_df(df):
    standardized = (df - df.mean()) / df.std()
    return standardized
