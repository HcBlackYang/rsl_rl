# # # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # # SPDX-License-Identifier: BSD-3-Clause
# # # 
# # # Redistribution and use in source and binary forms, with or without
# # # modification, are permitted provided that the following conditions are met:
# # #
# # # 1. Redistributions of source code must retain the above copyright notice, this
# # # list of conditions and the following disclaimer.
# # #
# # # 2. Redistributions in binary form must reproduce the above copyright notice,
# # # this list of conditions and the following disclaimer in the documentation
# # # and/or other materials provided with the distribution.
# # #
# # # 3. Neither the name of the copyright holder nor the names of its
# # # contributors may be used to endorse or promote products derived from
# # # this software without specific prior written permission.
# # #
# # # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# # #
# # # Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories

class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)

        super().__init__(num_actor_obs=rnn_hidden_size,
                         num_critic_obs=rnn_hidden_size,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std)

        activation = get_activation(activation)

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
    
    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0



# import numpy as np

# import torch
# import torch.nn as nn
# from torch.distributions import Normal
# from torch.nn.modules import rnn
# from .actor_critic import ActorCritic, get_activation
# from rsl_rl.utils import unpad_trajectories

# class ActorCriticRecurrent(ActorCritic):
#     is_recurrent = True
#     def __init__(self,  num_actor_obs,
#                         num_critic_obs,
#                         num_actions,
#                         num_envs, # <<<--- Keep num_envs parameter
#                         actor_hidden_dims=[256, 256, 256],
#                         critic_hidden_dims=[256, 256, 256],
#                         activation='elu',
#                         rnn_type='lstm',
#                         rnn_hidden_size=256,
#                         rnn_num_layers=1,
#                         init_noise_std=1.0,
#                         **kwargs):
#         # ... (init remains the same as previous fix) ...
#         if kwargs:
#             print("ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)

#         super().__init__(num_actor_obs=rnn_hidden_size,
#                          num_critic_obs=rnn_hidden_size,
#                          num_actions=num_actions,
#                          actor_hidden_dims=actor_hidden_dims,
#                          critic_hidden_dims=critic_hidden_dims,
#                          activation=activation,
#                          init_noise_std=init_noise_std)

#         activation = get_activation(activation)

#         self.memory_a = Memory(num_actor_obs, num_envs=num_envs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
#         self.memory_c = Memory(num_critic_obs, num_envs=num_envs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

#         print(f"Actor RNN: {self.memory_a}")
#         print(f"Critic RNN: {self.memory_c}")


#     def reset(self, dones=None):
#         self.memory_a.reset(dones)
#         self.memory_c.reset(dones)

#     def act(self, observations, masks=None, hidden_states=None):
#         # memory_a.forward returns (output, new_hidden_state)
#         input_a, _ = self.memory_a(observations, masks, hidden_states)
#         # --- FIX: Remove squeeze(0) ---
#         # Pass the direct output of the memory module to the base class act method
#         return super().act(input_a)
#         # -----------------------------

#     def act_inference(self, observations):
#         # memory_a.forward returns (output, new_hidden_state) in inference mode
#         # The output 'input_a' here has shape [BatchSize, HiddenSize]
#         input_a, _ = self.memory_a(observations)
#         # Pass the RNN output to the base class inference method
#         # Base class act_inference might expect unsqueezed input if it handles batches? Check ActorCritic.act_inference.
#         # Assuming base act_inference handles batch dimension correctly:
#         return super().act_inference(input_a)
#         # If base act_inference expects single item [HiddenSize], then maybe squeeze is needed, but unlikely.
#         # return super().act_inference(input_a.squeeze(0)) # Original problematic line for single item

#     def evaluate(self, critic_observations, masks=None, hidden_states=None):
#         # memory_c.forward returns (output, new_hidden_state)
#         input_c, _ = self.memory_c(critic_observations, masks, hidden_states)
#         # Pass the direct output of the memory module to the base class evaluate method
#         return super().evaluate(input_c) # Already removed squeeze here

#     def get_hidden_states(self):
#         return self.memory_a.hidden_states, self.memory_c.hidden_states

# # ... (Memory class remains the same as the previous corrected version) ...
# class Memory(torch.nn.Module):
#     # --- 添加 num_envs 参数 ---
#     def __init__(self, input_size, num_envs, type='lstm', num_layers=1, hidden_size=256):
#     # ---------------------------
#         super().__init__()
#         self.input_size = input_size
#         self.num_envs = num_envs # <<<--- 存储 num_envs
#         self.type = type.lower()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size

#         rnn_cls = nn.GRU if self.type == 'gru' else nn.LSTM
#         self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

#         self.is_lstm = self.type == 'lstm'
#         # --- 使用 num_envs 初始化 hidden_states ---
#         self.init_hidden_states() # 调用初始化方法
#         # ----------------------------------------
#         # print(f"Memory initialized: type={self.type}, layers={num_layers}, hidden={hidden_size}, num_envs={num_envs}")
#         # if self.hidden_states is not None:
#         #     if isinstance(self.hidden_states, tuple):
#         #         print(f"  Initial hidden state shape: h={self.hidden_states[0].shape}, c={self.hidden_states[1].shape}")
#         #     else:
#         #         print(f"  Initial hidden state shape: h={self.hidden_states.shape}")

#     def init_hidden_states(self, device='cpu'):
#         """Initializes hidden states with the correct batch size (num_envs)."""
#         if self.is_lstm:
#             # Shape: (num_layers, BatchSize=num_envs, hidden_size)
#             self.hidden_states = (torch.zeros(self.num_layers, self.num_envs, self.hidden_size, device=device),
#                                   torch.zeros(self.num_layers, self.num_envs, self.hidden_size, device=device))
#         else: # GRU
#             # Shape: (num_layers, BatchSize=num_envs, hidden_size)
#             self.hidden_states = torch.zeros(self.num_layers, self.num_envs, self.hidden_size, device=device)

#     def forward(self, input, masks=None, hidden_states=None):
#         # input shape: Batch Mode [SeqLen, BatchSize, InputSize] or Inference Mode [BatchSize, InputSize]
#         # masks shape: Batch Mode [SeqLen, BatchSize]
#         # hidden_states shape: Batch Mode [(NumLayers, BatchSize, HiddenSize), ...]

#         # --- 确保内部存储的 hidden_states 在正确的设备上 ---
#         if self.hidden_states is not None:
#             current_device = input.device
#             if isinstance(self.hidden_states, tuple): # LSTM
#                 if self.hidden_states[0].device != current_device:
#                     # print(f"DEBUG: Moving LSTM hidden states to {current_device}")
#                     self.hidden_states = (self.hidden_states[0].to(current_device), self.hidden_states[1].to(current_device))
#             else: # GRU
#                 if self.hidden_states.device != current_device:
#                     # print(f"DEBUG: Moving GRU hidden state to {current_device}")
#                     self.hidden_states = self.hidden_states.to(current_device)
#         # ------------------------------------------------------

#         batch_mode = masks is not None
#         if batch_mode:
#             # Batch mode (policy update): use passed hidden_states from storage
#             if hidden_states is None:
#                 raise ValueError("Hidden states not passed to memory module during policy update")

#             # Ensure passed hidden_states are on the correct device
#             if isinstance(hidden_states, tuple): # LSTM
#                 hidden_states = (hidden_states[0].to(input.device), hidden_states[1].to(input.device))
#             else: # GRU
#                 hidden_states = hidden_states.to(input.device)

#             # input shape [SeqLen, BatchSize, InputSize]
#             out, new_hidden_states_update = self.rnn(input, hidden_states)
#             # new_hidden_states_update are the states *after* processing the sequence

#             # --- FIX for unpad_trajectories ---
#             out_len = out.shape[0]
#             mask_len = masks.shape[0]
#             if mask_len > out_len:
#                 masks_for_unpad = masks[:out_len, ...]
#             elif mask_len < out_len:
#                 # print(f"Error: RNN output length ({out_len}) > mask length ({mask_len})")
#                 padding_size = mask_len - out_len
#                 out = torch.cat([out, torch.zeros(padding_size, *out.shape[1:], device=out.device, dtype=out.dtype)], dim=0)
#                 masks_for_unpad = masks
#             else:
#                 masks_for_unpad = masks
#             out_unpadded = unpad_trajectories(out, masks_for_unpad)
#             # ---------------------------------
#             # For batch mode, the returned hidden state is the state *after* the sequence
#             # Return the state computed by RNN after processing the batch sequence
#             return out_unpadded, new_hidden_states_update

#         else:
#             # Inference mode (collection): use internal hidden_states
#             # Input shape [BatchSize, InputSize], needs unsqueeze
#             # Ensure internal hidden state batch size matches input batch size
#             # This is important if envs reset during rollout
#             current_batch_size = input.shape[0]
#             if self.hidden_states is not None:
#                  state_batch_size = self.hidden_states[0].shape[1] if self.is_lstm else self.hidden_states.shape[1]
#                  if state_batch_size != current_batch_size:
#                     # This indicates an issue, likely need to call reset more carefully
#                     # print(f"Warning: Memory hidden state batch size {state_batch_size} != input batch size {current_batch_size}. Re-initializing hidden states.")
#                     self.init_hidden_states(device=input.device) # Re-init with correct size and device

#             # unsqueeze input for sequence length 1
#             out, new_hidden_states_inference = self.rnn(input.unsqueeze(0), self.hidden_states)
#             # Update the stored hidden state for the next step
#             self.hidden_states = new_hidden_states_inference
#             # out shape: [1, BatchSize, HiddenSize] -> squeeze sequence dim
#             out_unpadded = out.squeeze(0) # Shape [BatchSize, HiddenSize]
#             # Return the *updated* internal hidden state
#             return out_unpadded, self.hidden_states

    

#     def reset(self, dones=None):
#         if self.hidden_states is None:
#              # print("Warning: Trying to reset None hidden_states. Initializing.")
#              self.init_hidden_states() # Initialize if None
#              # Need to know the device, default to cpu? Or require caller to handle?
#              # Let's assume init_hidden_states handles default device or caller sets device later.
#              return

#         if dones is None:
#              # Reset all hidden states
#              if self.is_lstm:
#                  self.hidden_states = (self.hidden_states[0] * 0.0, self.hidden_states[1] * 0.0)
#              else:
#                  self.hidden_states = self.hidden_states * 0.0
#         else:
#              # Reset specific environment states using dones mask (ensure dones is on same device)
#              hidden_device = self.hidden_states[0].device if self.is_lstm else self.hidden_states.device
#              dones_device = dones.to(hidden_device)
#              # Ensure batch dim matches before applying mask
#              target_batch_dim = self.hidden_states[0].shape[1] if self.is_lstm else self.hidden_states.shape[1]
#              if dones_device.shape[0] == target_batch_dim:
#                  if self.is_lstm:
#                      h, c = self.hidden_states
#                      h[:, dones_device, :] = 0.0
#                      c[:, dones_device, :] = 0.0
#                      # No need to reassign tuple, modified in-place
#                  else:
#                      self.hidden_states[:, dones_device, :] = 0.0
#              # else:
#                  # print(f"Warning: Memory.reset dones shape {dones_device.shape} mismatch hidden state batch dim {target_batch_dim}. Skipping reset.")
