"""
NOTE: THIS FILE ONLY (lstm_encoder.py) is a significantly modified version of the source code in the cnn_lstm example of the torch multimodal library,
the modified file is: https://github.com/facebookresearch/multimodal/blob/main/examples/cnn_lstm/lstm_encoder.py

The original source code in the torch multimodal library for lstm_encoder.py is licensed under a BSD 3-Clause license,
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
from typing import Literal, List

import torch
import torch.nn as nn

from transformers import AutoModel


class LSTMEncoder(nn.Module):

    def __init__(self, model_name: str, model_hidden_states_num: int,
                 directions_fusion_strat: Literal["sum", "mean", "concat"] = "concat",
                 freeze_embedding_model: bool = False):

        super().__init__()

        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

        if freeze_embedding_model:
            for param in self.model.parameters():
                param.requires_grad = False

        # first value of tuple is the function and second is the expected output size of that function
        directions_available_fusions = {"sum": (self._fuse_directions_sum, self.model.config.hidden_size * 2),
                                        "mean": (self._fuse_directions_mean, self.model.config.hidden_size * 2),
                                        "concat": (self._fuse_directions_concat, self.model.config.hidden_size * 4)}

        self.directions_fusion_strat, self.expected_output_size = directions_available_fusions[directions_fusion_strat]

        self.lstm = nn.LSTM(
            input_size=self.model.config.hidden_size,
            hidden_size=self.model.config.hidden_size * 2,
            bidirectional=True,
            batch_first=True,
        )

        self.hidden_states_num = model_hidden_states_num

    def _fuse_directions_sum(self, directions: List[torch.Tensor]):
        return torch.stack(directions).sum(0)

    def _fuse_directions_mean(self, directions: List[torch.Tensor]):
        return torch.stack(directions).mean(0)

    def _fuse_directions_concat(self, directions: List[torch.Tensor]):
        return torch.concat(directions, dim=-1)

    def forward(self, x) -> torch.Tensor:

        embeddings = torch.stack(self.model(**x).hidden_states[-self.hidden_states_num:]).mean(0)

        _, (h, _) = self.lstm(embeddings)
        out = self.directions_fusion_strat([h[0, :, :], h[1, :, :]])

        return out
