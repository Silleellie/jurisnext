"""
NOTE: THIS FILE ONLY (cnn_encoder.py) is a modified version of the source code in the cnn_lstm example of the torch multimodal library,
the modified file is: https://github.com/facebookresearch/multimodal/blob/main/examples/cnn_lstm/cnn_encoder.py

The original source code in the torch multimodal library for cnn_encoder.py is licensed under a BSD 3-Clause license,
the following is a copy of the license text (copy required by the license itself):

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Meta nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):

    def __init__(self, input_dims, output_dims, kernel_sizes):

        super().__init__()

        layers = nn.ModuleList()
        leaky_relu = nn.LeakyReLU()
        max_pool2d = nn.MaxPool2d(2, stride=2)

        for in_channels, out_channels, kernel_size in zip(input_dims, output_dims, kernel_sizes):

            padding_size = kernel_size // 2

            conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding_size)
            batch_norm_2d = nn.BatchNorm2d(out_channels)

            layers.append(nn.Sequential(conv, leaky_relu, max_pool2d, batch_norm_2d))

        layers.append(nn.Flatten())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
