"""
Author: Vlad Timu (timuvlad@gmail.com)
Date: Aug 18 2024
Description: Vizualizer for the RSNA dataset
"""

import pretty_errors
from pydicom import read_file
import logging
from glob import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import rerun as rr
from argparse import ArgumentParser
import cv2 as cv

def setup_logging() -> None:
    logger = logging.getLogger()
    rerun_handler = rr.LoggingHandler("logs")
    rerun_handler.setLevel(logging.DEBUG)
    rerun_handler.setFormatter(logging.Formatter("%(asctime)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s"))
    logger.addHandler(rerun_handler)

RAW_DATA_ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "raw")

def parse_script_args():
    parser = ArgumentParser()
    parser.add_argument("-uid", "--study_uid", type=str, help="ID of the study to visualize", required=True)
    parser.add_argument("-s", "--split", type=str, help="Split of the dataset to visualize", default="train")
    parser.add_argument("-t", "--type", type=str, help="Type of data to visualize", choices=["raw", "annotated"], default="annotated")
    return parser.parse_args()

def main():
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel("DEBUG")
    setup_logging()

    args = parse_script_args()

    if args.split != "train":
        raise NotImplementedError(f"Visualization not implemented for split {args.s}")
    
    study_dir: list[str] = os.listdir(f"{RAW_DATA_ROOT_PATH}/train_images/{args.study_uid}")

    series_descriptions: pd.DataFrame = pd.read_csv(f"{RAW_DATA_ROOT_PATH}/train_series_descriptions.csv")
    series_paths: dict[str, str] = {x: f"{RAW_DATA_ROOT_PATH}/train_images/{args.study_uid}/{x}" for x in study_dir}
    study_data: dict[str, dict[str, str|list[str]]] = { study_uid: { 'FolderPath': study_path, 
                        'SOPInstanceNames': [] 
                    } 
                for study_uid, study_path in series_paths.items() }
    annotations: pd.DataFrame = pd.read_csv(f"{RAW_DATA_ROOT_PATH}/train_label_coordinates.csv")
    study_annotations: pd.DataFrame = annotations[annotations["study_id"] == int(args.study_uid)]

    for study_series in tqdm(study_data):
        images_paths: list[str] = list(
            filter(lambda x: x.find('.DS') == -1, 
                sorted(os.listdir(study_data[study_series]['FolderPath']), key=lambda x: int(x.split(".")[0]))
                )
        )
        series_annotations: pd.DataFrame = study_annotations[study_annotations["series_id"] == int(study_series)]
        study_data[study_series]['SOPInstanceNames'] = images_paths
        study_data[study_series]['SOPInstanceUIDs'] = [os.path.join(study_data[study_series]["FolderPath"], filename).split("/")[-1].replace(".dcm", "") for filename in images_paths]
        study_data[study_series]["Dicom"] = [read_file(os.path.join(study_data[study_series]["FolderPath"], filename)) for filename in images_paths]
        study_data[study_series]["RawPixels"] = np.array([dicom_file.pixel_array for dicom_file in study_data[study_series]["Dicom"]]).transpose(-1, 1, 0)
        study_data[study_series]['SeriesDescriptions'] = series_descriptions[
                (series_descriptions['study_id'] == int(args.study_uid)) & 
                (series_descriptions['series_id'] == int(study_series))
            ]['series_description'].iloc[0]
        study_data[study_series]["Annotations"] = []
        study_data[study_series]["AnnotatedPixels"] = np.zeros(study_data[study_series]["RawPixels"].shape, dtype=np.uint8)
        for sop_instance_uid, image_idx in zip(study_data[study_series]['SOPInstanceUIDs'], range(study_data[study_series]["RawPixels"].shape[-1])):
            study_data[study_series]["AnnotatedPixels"][:, :, image_idx] = cv.normalize(study_data[study_series]["RawPixels"][:, :, image_idx], None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            slice_annotations = series_annotations[series_annotations["instance_number"] == int(sop_instance_uid)]
            if not slice_annotations.empty:
                study_data[study_series]["Annotations"].append(slice_annotations[["condition", "level", "x", "y"]].reset_index(drop=True))
                for _, entry in slice_annotations.iterrows():
                    study_data[study_series]["AnnotatedPixels"][:, :, image_idx] = cv.circle(study_data[study_series]["AnnotatedPixels"][:, :, image_idx].copy(), (int(entry["y"]), int(entry["x"])), 10, (255, 0, 0), 1)

    rr.init("visualize_study", spawn=True)
    
    for series_id in study_data.keys():
        path = study_data[series_id]["SeriesDescriptions"]
        rr.log(
            entity_path=path,
            entity=rr.Tensor(
                data=np.rot90(study_data[series_id][f"{args.type.title()}Pixels"], k=3),
                dim_names=["height", "width", "depth"]
                )
        )
        
if __name__ == "__main__":
    main()
