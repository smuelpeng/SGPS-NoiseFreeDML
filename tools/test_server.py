from FeatureServer import NFClient2 as NFClient

feat_client = NFClient(1, 5775)
sample_indices, positive_indices = feat_client.get_groups_with_positive_indices(
            1, index=[1])
print(sample_indices, positive_indices)
