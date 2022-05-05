def confusion_matrix(TP, TN, FP, FN):
    """
    Tutorial 1.8
    """
    error_rate = (FP + FN) / (TP + TN + FP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2 * TP / (2 * TP + FP + FN)
    print(f"Error rate: {error_rate}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1-Score: {f1}")

    
if __name__ == '__main__':
    # Tutorial 1.8
    confusion_matrix(TP=3, TN=1, FP=1, FN=2)