# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
import torch
from ikomia import core, dataprocess, utils
import os
import sys
import argparse
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import gdown

from infer_modnet_portrait_matting.MODNet.MODNet.src.models.modnet import MODNet

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------


class InferModnetPortraitMattingParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)

        # Place default value initialization here
        self.model_name = "resnet34"
        self.cuda = torch.cuda.is_available()
        self.input_size = 1024
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = utils.strtobool(param_map["cuda"])
        self.input_size = int(param_map["input_size"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["cuda"] = str(self.cuda)
        param_map["input_size"] = str(self.input_size)
        return param_map

# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------


class InferModnetPortraitMatting(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        self.add_output(dataprocess.CImageIO())

        # Create parameters class
        if param is None:
            self.set_param_object(InferModnetPortraitMattingParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.model = None
        self.model_weight_url = 'https://drive.google.com/file/d/1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz/view?usp=drive_link'
        self.model_name = 'modnet_photographic_portrait_matting.ckpt'

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def combined_display(self, image, matte):
        if len(image.shape) == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]
        matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
        foreground = image * matte + np.full(image.shape, 255) * (1 - matte)
        foreground = cv2.resize(foreground, (image.shape[1], image.shape[0]))
        return foreground

    def run(self):
        # Core function of your process
        # Call begin_task_run() for initialization
        self.begin_task_run()
        # Get parameters :
        param = self.get_param_object()

        # Get input:
        input = self.get_input(0)

        # Get image from input/output (numpy array):
        input_image = input.get_image()

        # # define hyper-parameters
        # param.input_size = 512

        # define image to tensor transform
        im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        # Load model
        if param.update or self.model is None:
            print("loading pre-trained MODNet model ...")
            # Set path
            model_folder = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), "weights")
            model_weights = os.path.join(str(model_folder), self.model_name)

            # Download model if not exist
            if not os.path.isfile(model_weights):
                os.makedirs(model_folder, exist_ok=True)
                gdown.download(self.model_weight_url,
                               model_weights, quiet=False)

            # create MODNet and load the pre-trained ckpt
            self.model = MODNet(backbone_pretrained=False)
            self.model = nn.DataParallel(self.model)

            if param.cuda:
                self.model = self.model.cuda()
                weights = torch.load(model_weights)
            else:
                weights = torch.load(
                    model_weights, map_location=torch.device('cpu'))
            self.model.load_state_dict(weights)
            self.model.eval()
            param.update = False

        # inference image
        if len(input_image.shape) == 2:
            input_image = input_image[:, :, None]
        if input_image.shape[2] == 1:
            input_image = np.repeat(input_image, 3, axis=2)
        elif input_image.shape[2] == 4:
            input_image = input_image[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(input_image)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < param.input_size or min(im_h, im_w) > param.input_size:
            if im_w >= im_h:
                im_rh = param.input_size
                im_rw = int(im_w / im_h * param.input_size)
            elif im_w < im_h:
                im_rw = param.input_size
                im_rh = int(im_h / im_w * param.input_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(
            im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()

        bin_img = (matte * 255).astype('uint8')

        portrait = self.combined_display(input_image, bin_img)

        # Set output:
        output_bin = self.get_output(0)
        output_bin.set_image(bin_img)

        output = self.get_output(1)
        output.set_image(portrait)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferModnetPortraitMattingFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_modnet_portrait_matting"
        self.info.short_description = "Inference of MODNet Portrait Matting."
        self.info.description = "This algorithm proposes inference MODNet a Trimap-Free Portrait Matting."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Background"
        self.info.version = "1.0.0"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Zhanghan Ke and Jiayu Sun and Kaican Li and Qiong Yan and Rynson W.H. Lau"
        self.info.article = "MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition"
        self.info.journal = "AAAI"
        self.info.year = 2022
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/pdf/2011.11961.pdf"
        # Code source repository
        self.info.repository = "https://github.com/ZHKKKe/MODNet"
        # Keywords used for search
        self.info.keywords = "Portrait matting, Semantic segmentation, Trimap"

    def create(self, param=None):
        # Create process object
        return InferModnetPortraitMatting(self.info.name, param)
