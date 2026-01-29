def check_impersonation(predicted, claimed, confidence, threshold=0.65):
    if predicted != claimed and confidence >= threshold:
        return True
    return False
