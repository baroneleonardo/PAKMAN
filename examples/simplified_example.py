import sklearn.datasets


N_INITIAL_POINTS = 5
N_ITERATIONS = 5
N_POINTS_PER_ITERATION = 3  # The q- parameter
MODE = 'EI'  # 'EI' vs 'KG'

TARGET_COLUMN = 'target'

precomputed_sample_df = sklearn.datasets.load_diabetes(as_frame=True)['frame']

# No need for bounds, BUT you might need to specify column names for first derivative, second derivative, etc.
domain = PrecomputedDomain(dataset=precomputed_sample_df, target_column=TARGET_COLUMN)

