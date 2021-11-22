from sklearn.ensemble import GradientBoostingClassifier

def initialize_model():
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                        max_depth=1, random_state=0)
    return model