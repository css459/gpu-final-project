#include "grid-search.h"
#include "cuda_util.h"

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

// Method to check for CUDA errors
#define cudaCheckError(err) {                                                                    \
    if (err != cudaSuccess) {                                                                    \
        fprintf(stderr,"[ ERR ] CUDA: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                                                                 \
    }                                                                                            \
}

//
// helper to see if a value in array: used for accessing values in subsamples
//
int is_value_in_array(int val, int *arr, int size){
    int i;
    for (i=0; i < size; i++) {
        if (arr[i] == val)
            return 1;
    }
    return 0;
}

//
// Calculate the Gini Impurity for a particular split and given the target values
//
float calculate_gini_impurity(int** two_halves, float* target_values) {
  int n_instances = sizeof(two_halves) / sizeof(int);
  float gini_impurity = 0.0;
  for(int i=0; i < 2; i++) {
    int* half = two_halves[i];
    int len = sizeof(half) / sizeof(int);
    if(len != 0) {
      float score = 0.0;
      for(int j=0; j < sizeof(target_values)/sizeof(float); j++) {
        float target = (int) target_values[j];
        int count = 0;
        for(int row=0; row < len; row+=21) {
          int prediction = half[row];
          if(target == prediction) {
            count++;
          }
        }
        float p = count / len;
        score += p * p;
      }
      gini_impurity += (1.0 - score) * (len / n_instances);
    }
  }
  return gini_impurity;
}

//
// Make a leaf node out of two splits of data
//
int leaf_node(int* first, int* second) {
  int ones = 0;
  int zeroes = 0;
  int size_first = sizeof(first) / sizeof(int);
  for(int row=0; row < size_first; row+=21) {
    if(first[row] == 0) {
      zeroes++;
    } else {
      ones++;
    }
  }
  if(second != NULL) {
    int size_second = sizeof(second) / sizeof(int);
    for(int row=0; row < size_first; row+=21) {
      if(second[row] == 0) {
        zeroes++;
      } else {
        ones++;
      }
    }
  }
  if(ones >= zeroes) {
    return 1;
  } else {
    return 0;
  }
}

//
// Grow the tree by evaluating the best split and split into two halves
//
void grow(TreeNode* node, int depth, int max_depth, int min_size, int n_features) {
  int* left = node->group1;
  int* right = node->group2;

  node->group1 = NULL;
  node->group2 = NULL;

  if(left == NULL && right == NULL) {
    node->left_array = leaf_node(left, right);
    node->right_array = leaf_node(left, right);
    return;
  }
  if(depth >= max_depth) {
    node->left_array = leaf_node(left, NULL);
    node->right_array = leaf_node(right, NULL);
    return;
  }
  if(sizeof(left) / sizeof(int) <= min_size) {
    node->left_array = leaf_node(left, NULL);
  } else {
    node->left_node = split_data_based_on_features((float*)left, n_features);
    grow(node->left_node, max_depth, min_size, n_features, depth+1);
  }

  if(sizeof(right) / sizeof(int) <= min_size) {
    node->right_array = leaf_node(right, NULL);
  } else {
    node->right_node = split_data_based_on_features(right, n_features);
    grow(node->right_node, max_depth, min_size, n_features, depth+1);
  }
}

//
// Split the dataset into a test and training set
//
int** split_dataset_in_half(float **dataset, float value, int index, int elements_per_line) {
  int elements_in_dataset = sizeof(dataset) / sizeof(float);

  int** result = malloc(elements_in_dataset * sizeof(int));

  int* half1 = malloc(elements_in_dataset / 2 * sizeof(int));
  int* half2 = malloc(elements_in_dataset / 2 * sizeof(int));

  int index1 = 0;
  int index2 = 0;

  for(int row=0; row < sizeof(dataset) / sizeof(dataset[0]); row++) {
    if(dataset[row][index] < value) {
      // copy over the entire row moving up the index
      for(int i=0; i < elements_per_line; i++) {
        half1[index1] = dataset[row][i];
        index1++;
      }
    } else {
      // copy over the entire row moving up the index
      for(int i=0; i < elements_per_line; i++) {
        half2[index2] = dataset[row][i];
        index2++;
      }
    }
  }
  result[0] = half1;
  result[1] = half2;
  return result;
}

//
// Get the best point of split for data based on features
//
TreeNode* split_data_based_on_features(float** dataset, int n_features, int elements_per_line) {
  int number_rows = (sizeof(dataset) / sizeof(float*));

  float* target_values = malloc(number_rows * sizeof(float));
  for(int i=0; i < number_rows; i++) {
    target_values[i] = dataset[i][elements_per_line-1];
  }

  int b_index = INT_MAX;
  float b_value = INT_MAX;
  float b_score = INT_MAX;
  int* group1;
  int* group2;

  int* features = malloc(size_of(float) * n_features);
  int count_features = 0;
  while(count_features < n_features) {
    int random_feature_index = rand(elements_per_line-1);
    if(is_value_in_array(random_feature_index, features, size_of(float) * n_features) == 0) {
      features[count_features] = random_feature_index;
      count_features++;
    }
  }

  for(int i=0; i < n_features; i++) {
    int feature_index = features[i];
    for(int row=0; row < number_rows; row++) {
      float feature_value = dataset[row][feature_index];
      int** two_halves = split_dataset_in_half(dataset, feature_value, feature_index, elements_per_line);
      float gini_impurity = calculate_gini_impurity(two_halves, target_values);
      if(gini_impurity < b_score) {
        b_index = feature_index;
        b_value = feature_value;
        b_score = gini_impurity;
        group1 = two_halves[0];
        group2 = two_halves[1];
      }
    }
  }
  TreeNode* node = make_tree_node(b_index, b_value, group1, group2);
  return node;
}

//
// Build a single decision tree
//
TreeNode* build_tree(float* training_data, int max_depth, int min_size, int n_features) {
  TreeNode* root = split_data_based_on_features(training_data, n_features);
  grow(&root, max_depth, min_size, n_features, 1);
  return root;
}

//
// Calculate the percentage of accurate predictions - used to measure output of a particular Random Forest with particular hyper parameters
//
float calculate_accuracy(int n, int* actual, int* predicted) {
  int correct = 0;
  for(int i=0; i < n; i++) {
    if(actual[i] == predicted[i]) {
      correct += 1;
    }
  }
  return ((float) correct / (float) n) * 100.0;
}

//
// Predict based on one tree rooted at <node>
//
int predict(TreeNode* node, int row, float* test_data) {
  if(test_data[row][node->index] < node->value) {
    if(node->left_node != NULL) {
      return predict(node->left_node, row, test_data);
    } else {
      return node->left_array;
    }
  } else {
    if(node->right_node != NULL) {
      return predict(node->right_node, row, test_data);
    } else {
      return node->right_array;
    }
  }
}

//
// Run predictions for a set of grown trees and for a particular row in the data set and chose the prediction that most trees agree on
//
int aggregate_predictions(int n_trees, TreeNode* trees, int row, float* test_data) {
  int ones = 0;
  int zeroes = 0;
  for(int i=0; i < sizeof(trees) / sizeof(TreeNode); i++) {
    TreeNode* tree = trees[i];
    int prediction_of_this_tree = predict(tree, row, test_data);
    if(prediction_of_this_tree == 1) ones++;
    else zeroes++;
  }
  if(ones >= zeroes) {
    return 1;
  } else {
    return 0;
  }
}

__global__ void grow_random_forest(TreeNode* trees, float* training_data, float* test_data, int n_trees, int n_features, int max_depth, int min_size) {

}

//
// Grow a random forest for a particular set of parameters
//
int* random_forest(float** cuda_memory, int n_trees, int n_features, int max_depth, int min_size) {
  int deviceId;
  cudaGetDevice(&deviceId);

  float* data_for_device = cuda_memory[deviceId];

  struct TreeNode* trees = (TreeNode*)cudaMalloc(n_trees * sizeof(struct TreeNode*));

  err = cudaMemcpyAsync((void *) dest, (void *) src, memsize, cudaMemcpyHostToDevice, streams[i]);
  cudaCheckError(err);

  float** test_and_train = get_test_and_train(data_for_device);

  int* results = cudaMalloc(sizeof(int) * 3);
  grow_random_forest<<< >>>


  // grow the trees
  for(int i=0; i < n_trees; i++) {
    struct TreeNode* tree = build_tree(training_data, max_depth, min_size, n_features);
    trees[i] = tree;
  }

  int* predictions = (int*)malloc(n_trees * sizeof(int));
  // test the trees
  for(int row=0; row < sizeof(test_data) / sizeof(float); row++) {
    predictions[row] = aggregate_predictions(n_trees, trees, row, test_data);
  }
  return calculate_accuracy(n_trees*sizeof(int), predictions, test_and_train[1]);
}

//
// run grid search across multiple RF models
//
int* grid_search(float** cuda_memory, SamplingProperties* props) {
  // varying estimators, size, depth
  // note: number of features set to sqrt(features) to
  int[3] n_estimators = {10,100,1000};
  int[3] min_size = {10,100,100};
  int[3] max_depth = {10,100,1000};

  int* optimal_parameters = malloc(sizeof(int)*3);
  float best_accuracy = 0.0;

  // loop over combinations with streams
  for(int i=0; i < 3; i++) {
    for(int j=0; j < 3; j++) {
      for(int k=0; k < 3; k++) {
        int number_of_trees = n_estimators[i];
        int size = min_size[j];
        int maximum_depth = max_depth[k];
        float accuracy = random_forest(cuda_memory, number_of_trees, 5, maximum_depth, size);
        if(accuracy > best_accuracy) {
          best_accuracy = accuracy;
          // update
          optimal_parameters[0] = number_of_trees;
          optimal_parameters[1] = min_size;
          optimal_parameters[2] = max_depth;
        }
      }
    }
  }
  return optimal_parameters;
}
