import os
import pickle
import numpy as np
import argparse

def split_features(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over each subdirectory in the input directory
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        
        if os.path.isdir(subdir_path):
            features_path = os.path.join(subdir_path, 'features.pkl')
            
            if os.path.isfile(features_path):
                # Load the 2D numpy array from the features.pkl file
                with open(features_path, 'rb') as f:
                    features = pickle.load(f)
                
                # Create a corresponding subdirectory in the output directory
                output_subdir_path = os.path.join(output_dir, subdir)
                if not os.path.exists(output_subdir_path):
                    os.makedirs(output_subdir_path)
                
                # Split the 2D numpy array into 1D arrays and save each as a separate pkl file
                for idx, feature in enumerate(features):
                    output_file_path = os.path.join(output_subdir_path, f'feature_{idx}.pkl')
                    with open(output_file_path, 'wb') as f:
                        pickle.dump(feature, f)

def main():
    parser = argparse.ArgumentParser(description='Split 2D numpy arrays in features.pkl files into multiple 1D numpy arrays.')
    parser.add_argument('input_dir', type=str, help='Path to the input directory.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory.')

    args = parser.parse_args()
    split_features(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()
