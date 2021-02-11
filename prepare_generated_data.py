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
    # Transform phenotypic data into dataframe
    abide_pheno = pd.DataFrame(abide.phenotypic)

    try: # check if feature file already exists
        # load features
        feat_file_autism = os.path.join(output_dir, 'ABIDE_BASC064_features_autism.npz')
        feat_file_control = os.path.join(output_dir, 'ABIDE_BASC064_features_control.npz')
        X_features_autism = np.load(feat_file_autism)['a']
        X_features_control = np.load(feat_file_control)['a']
        print("Feature file found.")

    except: # if not, extract features
        X_features_autism = [] # To contain upper half of matrix as 1d array
        X_features_control = []  # To contain upper half of matrix as 1d array
        print("No feature file found. Extracting features...")

        for i,sub in enumerate(fmri_filenames):
            # extract the timeseries from the ROIs in the atlas
            time_series = masker.fit_transform(sub)
            # create a region x region correlation matrix
            correlation_matrix = correlation_measure.fit_transform([time_series])[0]
            # add to our container
            if(abide_pheno['DX_GROUP'][i] == 1):
                X_features_autism.append(correlation_matrix)
            else:
                X_features_control.append(correlation_matrix)
            # keep track of status
            print('finished extracting %s of %s'%(i+1,len(fmri_filenames)))
        # Save features
        np.savez_compressed(os.path.join(output_dir, 'ABIDE_BASC064_features_autism'),a = X_features_autism)
        np.savez_compressed(os.path.join(output_dir, 'ABIDE_BASC064_features_control'),a = X_features_control)

    print("Loading Data...")


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
    prepare_data(args.data_dir, args.output_dir)


if __name__ == "__main__":
    run()