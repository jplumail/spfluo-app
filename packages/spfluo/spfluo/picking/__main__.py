import argparse
import csv
import os
import pickle
import shutil
from glob import glob
from typing import Sequence

import numpy as np

from .modules.annotating import extract_annotations
from .modules.annotating.extract import create_annotation_image
from .modules.posttraining import (
    evaluate_picking,
    evaluate_tilt,
    postprocess,
    predict_picking,
    predict_tilt,
)
from .modules.pretraining import crop, prepare
from .modules.training import train_picking, train_tilt
from .modules.utils import load_annotations, seed_all, send_mail

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable tensorflow INFO and WARNING


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Setup pipeline")
    # - General
    all_stages = [
        "extract",
        "prepare",
        "train",
        "predict",
        "postprocess",
        "evaluate",
        "crop",
    ]
    parser.add_argument("--stages", type=str, default=all_stages, nargs="+")
    parser.add_argument(
        "--task", type=str, default="picking", choices=["picking", "tilt"]
    )
    parser.add_argument("--rootdir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=None, nargs="+")
    parser.add_argument("--extension", type=str, default="npz")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--mail", action="store_true")
    # 0 - extract
    parser.add_argument("--slicer3d_dir", type=str, default=None)
    parser.add_argument("--downscale", type=float, default=1.0)
    # I - prepare
    parser.add_argument("--template_pointcloud", type=str, default=None)
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--make_u_masks", action="store_true")
    parser.add_argument("--train_test_split", type=float, default=None)
    parser.add_argument("--train_val_split", type=float, default=None)
    parser.add_argument("--crop_output_dir", type=str, default=None)
    parser.add_argument("--margin", type=int, default=0, nargs="+")
    parser.add_argument("--pos_ratio", type=float, default=1.0)
    parser.add_argument("--positive_only", action="store_true")
    # II - train
    #   1. general setup
    parser.add_argument("--mode", type=str, default="pu", choices=["pu", "fs"])
    parser.add_argument("--epoch_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    #   2. model setup
    parser.add_argument("--network", type=str, default="efficientnet-s")
    parser.add_argument("--width_factor", type=float, default=1.0)
    parser.add_argument("--depth_factor", type=float, default=1.0)
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--swa_lr", type=float, default=1e-5)
    #   3. data setup
    parser.add_argument("--num_pos_samples", type=int, default=None)
    parser.add_argument("--augment", type=float, default=0.0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--selected_particles", type=str, default=None)

    #   4. pu
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--num_particles_per_image", type=int, default=None)
    parser.add_argument("--load_u_masks", action="store_true")
    #   5. autoencoder
    parser.add_argument("--ae", action="store_true")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--learn_eta", action="store_true")
    parser.add_argument("--num_features", type=int, default=(8, 16, 32), nargs=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    # III - predict
    parser.add_argument("--testdir", type=str, default=None)
    parser.add_argument("--testset_size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--stride", type=int, default=None, nargs="+")
    parser.add_argument("--predict_on_u_mask", action="store_true")
    # IV - postprocess
    parser.add_argument("--predictions", type=str, default=None)
    parser.add_argument("--nms_threshold", type=float, default=0.1)
    parser.add_argument("--iterative", action="store_true")
    parser.add_argument("--last_step", type=str, default="spatial_average")
    parser.add_argument("--otsu_margin", type=float, default=0.5)
    # V - evaluate
    parser.add_argument("--step_to_evaluate", type=str, default="last_step")
    parser.add_argument("--make_labelled_volumes", action="store_true")
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument("--evaluation_threshold", type=float, default=0.2)
    parser.add_argument("--image_name", type=str, default=None)
    # Tilt args
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument(
        "--crop_dir", type=str, default=None
    )  # cropped particules to take for pretraining
    return parser.parse_args()


def arg_to_tuple_if_needed(arg: Sequence[int]) -> Sequence[int]:
    if isinstance(arg, int):
        arg = 3 * (arg,)
    if isinstance(arg, list) and len(arg) == 1:
        arg = 3 * (arg[0],)
    return arg


def guess_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.testdir is None and args.rootdir is not None and "predict" in args.stages:
        args.testdir = os.path.join(args.rootdir, "test")
    if args.checkpoint is None and "predict" in args.stages:
        basename = "checkpoint_swa" if args.swa else "checkpoint"
        args.checkpoint = os.path.join(args.output_dir, f"{basename}.pt")
    if args.predictions is None and (
        "postprocess" in args.stages or "evaluate" in args.stages
    ):
        args.predictions = os.path.join(args.output_dir, "predictions.pickle")
    if args.annotations is None and "evaluate" in args.stages:
        args.annotations = os.path.join(args.testdir, "test_coordinates.csv")
    return args


def mail_at_the_end(args: argparse.Namespace) -> None:
    name = args.output_dir.strip("/").split("/")[-1]
    body = f"""
    Job...: {name}
    Task..: {args.task}
    Stages: {', '.join(args.stages)}
    Pipeline finished.
    """
    send_mail("Pipeline", body)


def main(args: argparse.Namespace) -> None:
    # if shuffle is True, we want annotations shuffling to be unique for each run,
    # but the training must be deterministic, so we seed everything but numpy.
    seed_all(seed_numpy=not args.shuffle)
    args = guess_paths(args)
    args.patch_size = arg_to_tuple_if_needed(args.patch_size)
    args.stride = arg_to_tuple_if_needed(args.stride)
    args.margin = arg_to_tuple_if_needed(args.margin)
    if "extract" in args.stages:
        extract_annotations(
            args.slicer3d_dir,
            args.rootdir,
            args.output_dir,
            args.patch_size,
            args.downscale,
        )
    if "prepare" in args.stages:
        prepare(
            args.rootdir,
            args.extension,
            args.template_pointcloud,
            args.dataset_size,
            args.make_u_masks,
            args.train_test_split,
            args.train_val_split,
            args.crop_output_dir,
            args.patch_size,
            args.margin,
            args.pos_ratio,
            args.positive_only,
            args.downscale,
        )
    # + -------------------------------------- PICKING -------------------------------------- + #
    if "train" in args.stages and args.task == "picking":
        train_picking(
            args.mode,
            args.rootdir,
            args.batch_size,
            args.num_workers,
            args.output_dir,
            args.num_epochs,
            args.lr,
            args.network,
            args.width_factor,
            args.depth_factor,
            args.swa,
            args.swa_lr,
            args.patch_size,
            args.num_pos_samples,
            args.shuffle,
            args.augment,
            args.dim,
            args.extension,
            args.epoch_size,
            args.downscale,
            # pu optional params
            args.radius,
            args.num_particles_per_image,
            args.load_u_masks,
            args.ae,
            args.eta,
            args.learn_eta,
            args.num_features,
            args.hidden_dim,
        )
    if "predict" in args.stages and args.task == "picking":
        predict_picking(
            args.network,
            args.checkpoint,
            args.testdir,
            args.predict_on_u_mask,
            args.patch_size,
            args.dim,
            args.stride,
            args.batch_size,
            args.extension,
            args.testset_size,
            args.image_name,
            args.output_dir,
            args.downscale,
        )
    if "postprocess" in args.stages and args.task == "picking":
        postprocess(
            args.predictions,
            args.testdir,
            args.patch_size,
            args.stride,
            args.nms_threshold,
            args.iterative,
            args.last_step,
            args.otsu_margin,
            args.extension,
            args.testset_size,
            args.image_name,
            args.output_dir,
            args.downscale,
        )
    if "evaluate" in args.stages and args.task == "picking":
        evaluate_picking(
            args.predictions,
            args.annotations,
            args.patch_size,
            args.evaluation_threshold,
            args.step_to_evaluate,
            args.make_labelled_volumes,
            args.testdir,
            args.image_name,
            args.output_dir,
        )
    # + --------------------------------------- TILT ---------------------------------------- + #
    if "crop" in args.stages:
        raw_dir = os.path.join(args.rootdir, "raw")

        os.makedirs(raw_dir, exist_ok=True)
        for image_path in glob(os.path.join(args.slicer3d_dir, "*/*.tif")):
            image_name = os.path.basename(image_path)
            shutil.copyfile(image_path, os.path.join(raw_dir, image_name))

        _, ext = os.path.splitext(args.predictions)
        if ext == ".pickle":
            with open(args.predictions, "rb") as f:
                predictions = pickle.load(f)

            with open(os.path.join(raw_dir, "predictions.csv"), "w") as f:
                for image_name in predictions:
                    pred = predictions[image_name]["last_step"]
                    centers = np.stack(
                        [
                            (pred[:, 3] + pred[:, 0]) / 2,
                            (pred[:, 4] + pred[:, 1]) / 2,
                            (pred[:, 5] + pred[:, 2]) / 2,
                        ],
                        axis=1,
                    )
                    centers *= args.downscale
                    centers = np.rint(centers).astype(int)
                    for i, center in enumerate(centers):
                        f.write(image_name)
                        f.write(",")
                        f.write(str(i))
                        f.write(",")
                        f.write(str(center[2]))
                        f.write(",")
                        f.write(str(center[1]))
                        f.write(",")
                        f.write(str(center[0]))
                        f.write("\n")

            shutil.copyfile(
                os.path.join(args.rootdir, "coordinates.csv"),
                os.path.join(raw_dir, "annotations.csv"),
            )
            annotations = load_annotations(os.path.join(raw_dir, "annotations.csv"))
            annotations[:, 2:] = args.downscale * annotations[:, 2:].astype(float)
            with open(os.path.join(raw_dir, "annotations.csv"), "w", newline="") as f:
                csv.writer(f).writerows(annotations)

            crop(
                raw_dir,
                "cropped",
                crop_size=args.patch_size,
                margin=0,
                pos_ratio=1,
                positive_only=True,
                extension=args.extension,
                csv_name="predictions",  # picking predictions
            )

            create_annotation_image(
                raw_dir,
                os.path.join(raw_dir, "annotations.csv"),
                args.patch_size,
                (0, 255, 0),
                args.extension,
            )
            create_annotation_image(
                raw_dir,
                os.path.join(raw_dir, "predictions.csv"),
                args.patch_size,
                (255, 0, 0),
                args.extension,
            )

    if "train" in args.stages and args.task == "tilt":
        train_tilt(
            args.rootdir,
            args.batch_size,
            args.num_workers,
            args.output_dir,
            args.extension,
            args.selected_particles,
            args.crop_dir,
            args.num_epochs,
            args.lr,
            args.network,
            args.width_factor,
            args.depth_factor,
            args.swa,
            args.swa_lr,
            args.patch_size,
            args.num_pos_samples,
            args.shuffle,
            args.augment,
            args.epoch_size,
            args.ae,
            args.eta,
            args.learn_eta,
            args.num_features,
            args.hidden_dim,
            args.pretrain,
        )
    if "predict" in args.stages and args.task == "tilt":
        predict_tilt(
            args.rootdir,
            args.network,
            args.checkpoint,
            args.crop_dir,
            args.dim,
            args.batch_size,
            args.extension,
            args.testset_size,
            args.image_name,
            args.output_dir,
        )
    if "evaluate" in args.stages and args.task == "tilt":
        evaluate_tilt(
            args.predictions,
            args.annotations,
            args.image_name,
            args.output_dir,
            args.selected_particles,
        )
    if args.mail:
        mail_at_the_end(args)


if __name__ == "__main__":
    main(parse_args())
