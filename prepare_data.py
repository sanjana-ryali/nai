#!/usr/bin/env python

from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from argparse import ArgumentParser
import numpy as np
from sklearn.decomposition import PCA
import os
import pandas as pd

def prepare_data(data_dir, output_dir, pipeline = "cpac", quality_checked = True):
    # get dataset
    print("Loading dataset...")
    abide = datasets.fetch_abide_pcp(data_dir = data_dir,
                                     pipeline = pipeline,
                                     quality_checked = quality_checked)
    # make list of filenames
    fmri_filenames = abide.func_preproc

    # load atlas
    multiscale = datasets.fetch_atlas_basc_multiscale_2015()
    atlas_filename = multiscale.scale064

    # initialize masker object
    masker = NiftiLabelsMasker(labels_img=atlas_filename,
                               standardize=True,
                               memory='nilearn_cache',
                               verbose=0)

    # initialize correlation measure
    correlation_measure = ConnectivityMeasure(kind='correlation', vectorize=True,
                                             discard_diagonal=True)

    try: # check if feature file already exists
        # load features
        feat_file = os.path.join(output_dir, 'ABIDE_BASC064_features.npz')
        X_features = np.load(feat_file)['a']
        print("Feature file found.")

    except: # if not, extract features
        X_features = [] # To contain upper half of matrix as 1d array
        print("No feature file found. Extracting features...")

        for i,sub in enumerate(fmri_filenames):
            # extract the timeseries from the ROIs in the atlas
            time_series = masker.fit_transform(sub)
            # create a region x region correlation matrix
            correlation_matrix = correlation_measure.fit_transform([time_series])[0]
            # add to our container
            X_features.append(correlation_matrix)
            # keep track of status
            print('finished extracting %s of %s'%(i+1,len(fmri_filenames)))
        # Save features
        np.savez_compressed(os.path.join(output_dir, 'ABIDE_BASC064_features'),
                                         a = X_features)

    # Dimensionality reduction of features with PCA
    print("Running PCA...")
    pca = PCA(0.99).fit(X_features) # keeping 99% of variance
    X_features_pca = pca.transform(X_features)

    # Transform phenotypic data into dataframe
    abide_pheno = pd.DataFrame(abide.phenotypic)

    # Get the target vector
    y_target = abide_pheno['DX_GROUP']

    return(X_features_pca, y_target)

def fetch_generated_data(data_dir, output_dir):

    print("Loading dataset...")
    abide = datasets.fetch_abide_pcp(data_dir=data_dir,
                                     pipeline='cpac',
                                     quality_checked=True)

    feat_file = os.path.join(output_dir, 'ABIDE_BASC064_features.npz')
    X_features = np.load(feat_file)['a']

    feat_file_autism = os.path.join(output_dir, 'generated_data_asd_300.csv')
    X_features_autism = pd.read_csv(feat_file_autism, sep=',', header=None, skiprows=1)
    X_features_autism = X_features_autism.drop(X_features_autism.columns[0], axis=1)

    feat_file_control = os.path.join(output_dir, 'generated_data_tc_300.csv')
    X_features_control = pd.read_csv(feat_file_control, sep=',', header=None, skiprows=1)
    X_features_control = X_features_control.drop(X_features_control.columns[0], axis=1)

    X_features_all = np.vstack((X_features, X_features_autism.values, X_features_control.values))
    print('Stacked synthetic and original features ', X_features_all.shape)

    print("Running PCA...")
    pca = PCA(0.99).fit(X_features_all)  # keeping 99% of variance
    X_features_pca = pca.transform(X_features_all)

    # Transform phenotypic data into dataframe
    abide_pheno = pd.DataFrame(abide.phenotypic)

    # Get the target vector
    y_target = abide_pheno['DX_GROUP']

    # add dx_group for synthetic data
    y_target_arr = np.append(y_target.values, np.ones(100)) # add asd group
    y_target_arr = np.append(y_target_arr, np.zeros(100)) # add tc group

    y_target = pd.Series(y_target_arr)

    return (X_features_pca, y_target)

def run():
    description = "Prepare data for classifier on the ABIDE data to predict autism"
    parser = ArgumentParser(__file__, description)
    parser.add_argument("data_dir", action = "store",
                        help = """Path to the data directory that contains the
                        ABIDE data set. If you already have the data set, this
                        should be the folder that contains the subfolder
                        'ABIDE_pcp'. If this folder does not exists yet, it will
                        be created in the directory you provide.""")
    parser.add_argument("output_dir", action = "store",
                        help = """Path to the directory where you want to store
                        outputs.""")
    args = parser.parse_args()
    X_features_pca, y_target = prepare_data(args.data_dir, args.output_dir)

    return(X_features_pca, y_target)


if __name__ == "__main__":
    run()