#ifndef GPU_FINAL_PROJECT_FOREST_H
#define GPU_FINAL_PROJECT_FOREST_H

//
// Struct representing a decision tree node
//
struct TreeNode {
  TreeNode* left_node;
  TreeNode* right_node;
  int left_array;
  int right_array;

  int index;
  float value;
  int* group1;
  int* group2;
};

//
// Constructor for the TreeNode struct
//
TreeNode make_tree_node(int index, float value, int* group1, int* group2);

//
// Calculate the percentage of accurate predictions
//
float calculate_accuracy(int n, int* actual, int* predicted);

//
// Grow a random forest for a particular set of parameters
//
int* random_forest(float* training_data, float* test_data, int n_trees, int n_features, int max_depth, int min_size);

//
// Run predictions for a set of grown trees and for a particular row in the data set and chose the prediction that most trees agree on
//
int aggregate_predictions(TreeNode* trees, int row);

//
// Build a single decision tree
//
TreeNode build_tree(float* training_data, int max_depth, int min_size, int n_features);

//
// Get the best point of split for data based on features
//
TreeNode split_data_based_on_features(float **dataset, int n_features);

//
// Split the dataset into a test and training set
//
float** split_dataset_in_half(float **dataset, float value, int index, int elements_per_line);

//
// Grow the tree by evaluating the best split and split into two halves
//
void grow(TreeNode node, int depth, int max_depth, int min_size, int n_features);

//
// Create leaf nodes out of tree node sets
//
int leaf_node(int* first, int* second);

//
// Run grid search across multiple RF models
//
int* grid_search(float **dataset, SamplingProperties *props);

#endif
