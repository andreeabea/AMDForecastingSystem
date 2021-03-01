from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from time_series_analysis import RetinaFeatureAnalysis
from visual_acuity_analysis import VisualAcuityAnalysis

def get_features_labels(sequences):
    va_labels = []
    for i in range(len(sequences)):
        # get va
        va_labels.append(sequences[i][-1])

va_analysis = VisualAcuityAnalysis()
eyeData = va_analysis.get_va_df()

retina_analysis = RetinaFeatureAnalysis()
sequences = retina_analysis.get_feature_sequences(RetinaFeatureAnalysis.CENTRAL_AVG_THICKNESS_FEATURE, 0, eyeData, include_va=True)

if isinstance(sequences[0][0], float):
    nb_features = 1
else:
    nb_features = len(sequences[0][0])

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.25, random_state=42)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=100, bootstrap=True, random_state=42)
