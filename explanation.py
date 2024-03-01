from collections import Counter
from matplotlib import pyplot as plt

def explanation(reason):
    """
    This function takes the reason list and plots the most common reasons for the model's predictions.
    """
    # Flatten the reason list
    features = [j[0] for r in reason for j in r]
    feature_counts = Counter(features)
    # Sort features by their counts
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1])
    sorted_feature_names = [feature[0] for feature in sorted_features]
    sorted_feature_counts = [feature[1] for feature in sorted_features]

    # Plot the sorted value counts
    plt.figure(figsize=(10, 16))
    plt.barh(sorted_feature_names, sorted_feature_counts, color='skyblue')
    plt.xlabel('Count')
    plt.ylabel('Feature')
    plt.title('Sorted Feature Counts')

    for i, count in enumerate(sorted_feature_counts):
        plt.text(count, i, str(count), ha='left', va='center', color='black')

    plt.savefig('most_common_reasons.png', bbox_inches='tight')
    # plt.show()
    
    