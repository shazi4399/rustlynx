import pandas as pd
import numpy as np

BITLENGTH = 64
PRECISION = 10

def float_to_fixed_point(x, bitlength=BITLENGTH, precision=PRECISION):
    if x >= 0.0:
        return int(x * (1 << precision))
    return (1 << bitlength) - int((-x) * (1 << precision))

""" Load and Transform """
alice = pd.read_csv("./raw/alice.csv")
bob = pd.read_csv("./raw/bob.csv")
df = pd.concat([alice, bob], ignore_index=True)
df.drop('patient_id', axis=1, inplace=True)
df['cohort_type'] = df['cohort_type'].apply(lambda x: 1 if x == 'Wild Type' else 0)
df = df.apply(lambda x: pd.Series(list(map(float_to_fixed_point, x))))

""" Secret Share """
share0 = pd.DataFrame(np.random.randint(0, (1<<BITLENGTH)-1, size=df.shape, dtype=np.uint64),columns=list(df))
share1 = (df - share0) % (1 << BITLENGTH)
share1 = share1.apply(lambda x : pd.to_numeric(x, downcast='unsigned'))

share0 = pd.DataFrame(np.random.randint(0, (1<<BITLENGTH)-1, size=(128, 1876), dtype=np.uint64), columns=list(df))
share1 = pd.DataFrame(np.random.randint(0, (1<<BITLENGTH)-1, size=(128, 1876), dtype=np.uint64), columns=list(df))

share0['cohort_type'].to_csv("./secret-shared/class0.csv", header=False, index=False)
share1['cohort_type'].to_csv("./secret-shared/class1.csv", header=False, index=False)
share0.drop('cohort_type', axis=1, inplace=True)
share1.drop('cohort_type', axis=1, inplace=True)
share0.to_csv("./secret-shared/data0.csv", header=False, index=False)
share1.to_csv("./secret-shared/data1.csv", header=False, index=False)
