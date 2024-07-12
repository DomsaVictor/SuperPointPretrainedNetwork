# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use


from PIL import Image
import torch
from os import path
import numpy as np

from demo_superpoint import SuperPointNet, SuperPointFrontend

# Kapture is a pivot file format, based on text and binary files, used to describe SfM (Structure From Motion)
# and more generally sensor-acquired data
# it can be installed with
# pip install kapture
# for more information check out https://github.com/naver/kapture

import kapture
from kapture.io.records import get_image_fullpath
from kapture.io.csv import kapture_from_dir
from kapture.io.csv import get_feature_csv_fullpath, keypoints_to_file, descriptors_to_file
from kapture.io.features import get_keypoints_fullpath, keypoints_check_dir, image_keypoints_to_file
from kapture.io.features import get_descriptors_fullpath, descriptors_check_dir, image_descriptors_to_file
from kapture.io.csv import get_all_tar_handlers


def extract_kapture_keypoints(args):
    """
    Extract SuperPoint keypoints and descritors to the kapture format directly
    """
    print('extract_kapture_keypoints...')
    with get_all_tar_handlers(args.kapture_root,
                              mode={kapture.Keypoints: 'a',
                                    kapture.Descriptors: 'a',
                                    kapture.GlobalFeatures: 'r',
                                    kapture.Matches: 'r'}) as tar_handlers:
        kdata = kapture_from_dir(args.kapture_root, None,
                                 skip_list=[kapture.GlobalFeatures,
                                            kapture.Matches,
                                            kapture.Points3d,
                                            kapture.Observations],
                                 tar_handlers=tar_handlers)

        assert kdata.records_camera is not None
        image_list = [filename for _, _, filename in kapture.flatten(kdata.records_camera)]
        if args.keypoints_type is None:
            args.keypoints_type = path.splitext(path.basename(args.model))[0]
            print(f'keypoints_type set to {args.keypoints_type}')
        if args.descriptors_type is None:
            args.descriptors_type = path.splitext(path.basename(args.model))[0]
            print(f'descriptors_type set to {args.descriptors_type}')

        if kdata.keypoints is not None and args.keypoints_type in kdata.keypoints \
                and kdata.descriptors is not None and args.descriptors_type in kdata.descriptors:
            print('detected already computed features of same keypoints_type/descriptors_type, resuming extraction...')
            image_list = [name
                          for name in image_list
                          if name not in kdata.keypoints[args.keypoints_type] or
                          name not in kdata.descriptors[args.descriptors_type]]

        if len(image_list) == 0:
            print('All features were already extracted')
            return
        else:
            print(f'Extracting SuperPoint features for {len(image_list)} images')

        iscuda = 'cuda' if torch.cuda.is_available() else 'cpu'


        if kdata.keypoints is None:
            kdata.keypoints = {}
        if kdata.descriptors is None:
            kdata.descriptors = {}

        if args.keypoints_type not in kdata.keypoints:
            keypoints_dtype = None
            keypoints_dsize = None
        else:
            keypoints_dtype = kdata.keypoints[args.keypoints_type].dtype
            keypoints_dsize = kdata.keypoints[args.keypoints_type].dsize
        if args.descriptors_type not in kdata.descriptors:
            descriptors_dtype = None
            descriptors_dsize = None
        else:
            descriptors_dtype = kdata.descriptors[args.descriptors_type].dtype
            descriptors_dsize = kdata.descriptors[args.descriptors_type].dsize

        sp_frontend = SuperPointFrontend(weights_path=args.model,
                                         nms_dist=args.nms_dist,
                                         conf_thresh=args.conf_thresh,
                                         nn_thresh=args.nn_thresh,
                                         cuda=iscuda)


        for image_name in image_list:
            img_path = get_image_fullpath(args.kapture_root, image_name)
            print(f"\nExtracting features for {img_path}")
            img = Image.open(img_path).convert('L')

            img = np.array(img, dtype=np.float32) / 255.0
            width, height = img.shape

            resized_image = img
            resized_width = width
            resized_height = height

            # max_edge = args.max_edge
            # max_sum_edges = args.max_sum_edges
            # if max(resized_width, resized_height) > max_edge:
            #     scale_multiplier = max_edge / max(resized_width, resized_height)
            #     resized_width = math.floor(resized_width * scale_multiplier)
            #     resized_height = math.floor(resized_height * scale_multiplier)
            #     resized_image = img.resize((resized_width, resized_height))
            # if resized_width + resized_height > max_sum_edges:
            #     scale_multiplier = max_sum_edges / (resized_width + resized_height)
            #     resized_width = math.floor(resized_width * scale_multiplier)
            #     resized_height = math.floor(resized_height * scale_multiplier)
            #     resized_image = img.resize((resized_width, resized_height))



            # if iscuda:
            #     resized_image = resized_image.cuda()

            # extract keypoints/descriptors for a single image
            xys, desc, heatmap = sp_frontend.run(resized_image)

            xys = np.transpose(xys, (1, 0))
            desc = np.transpose(desc, (1, 0))

            if keypoints_dtype is None or descriptors_dtype is None:
                keypoints_dtype = xys.dtype
                descriptors_dtype = desc.dtype

                keypoints_dsize = xys.shape[1]
                descriptors_dsize = desc.shape[1]

                kdata.keypoints[args.keypoints_type] = kapture.Keypoints('spp', keypoints_dtype, keypoints_dsize)
                kdata.descriptors[args.descriptors_type] = kapture.Descriptors('spp', descriptors_dtype,
                                                                               descriptors_dsize,
                                                                               args.keypoints_type, 'L2')
                keypoints_config_absolute_path = get_feature_csv_fullpath(kapture.Keypoints,
                                                                          args.keypoints_type,
                                                                          args.kapture_root)
                descriptors_config_absolute_path = get_feature_csv_fullpath(kapture.Descriptors,
                                                                            args.descriptors_type,
                                                                            args.kapture_root)
                keypoints_to_file(keypoints_config_absolute_path, kdata.keypoints[args.keypoints_type])
                descriptors_to_file(descriptors_config_absolute_path, kdata.descriptors[args.descriptors_type])
            else:
                assert kdata.keypoints[args.keypoints_type].dtype == xys.dtype
                assert kdata.descriptors[args.descriptors_type].dtype == desc.dtype
                assert kdata.keypoints[args.keypoints_type].dsize == xys.shape[1]

                assert kdata.descriptors[args.descriptors_type].dsize == desc.shape[1]
                assert kdata.descriptors[args.descriptors_type].keypoints_type == args.keypoints_type
                assert kdata.descriptors[args.descriptors_type].metric_type == 'L2'

            keypoints_fullpath = get_keypoints_fullpath(args.keypoints_type, args.kapture_root,
                                                        image_name, tar_handlers)
            print(f"Saving {xys.shape[0]} keypoints to {keypoints_fullpath}")
            image_keypoints_to_file(keypoints_fullpath, xys)
            kdata.keypoints[args.keypoints_type].add(image_name)

            descriptors_fullpath = get_descriptors_fullpath(args.descriptors_type, args.kapture_root,
                                                            image_name, tar_handlers)
            print(f"Saving {desc.shape[0]} descriptors to {descriptors_fullpath}")
            image_descriptors_to_file(descriptors_fullpath, desc)
            kdata.descriptors[args.descriptors_type].add(image_name)

        if not keypoints_check_dir(kdata.keypoints[args.keypoints_type], args.keypoints_type,
                                   args.kapture_root, tar_handlers) or \
                not descriptors_check_dir(kdata.descriptors[args.descriptors_type], args.descriptors_type,
                                          args.kapture_root, tar_handlers):
            print('local feature extraction ended successfully but not all files were saved')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        "Extract SuperPoint local features for all images in a dataset stored in the kapture format")
    parser.add_argument("--model", type=str, required=True, help='model path')
    parser.add_argument('--keypoints-type', default=None,  help='keypoint type_name, default is filename of model')
    parser.add_argument('--descriptors-type', default=None,  help='descriptors type_name, default is filename of model')

    parser.add_argument("--kapture-root", type=str, required=True, help='path to kapture root directory')

    parser.add_argument("--scale-f", type=float, default=2**0.25)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)

    parser.add_argument("--nms_dist", type=int, default=4)
    parser.add_argument("--conf_thresh", type=float, default=0.015)
    parser.add_argument("--nn_thresh", type=float, default=0.7)

    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='use -1 for CPU')
    args = parser.parse_args()

    extract_kapture_keypoints(args)