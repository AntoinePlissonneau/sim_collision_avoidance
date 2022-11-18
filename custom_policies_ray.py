# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:58:32 2022

@author: aplissonneau

In this file can be defined any policy function architecture
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement

from ray.rllib.models.modelv2 import restore_original_dimensions

class LowresCNN(TorchModelV2, nn.Module):
    ''' 2D CNN architecture using an occupancy grid map of size [10,70] and 
        2 additionals scalar features as input.
    '''
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)
            self.conv1 = nn.Conv2d(3, 32, 3)
            nn.init.kaiming_normal_(self.conv1.weight)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3)
            nn.init.kaiming_normal_(self.conv2.weight)
            self.fc1 = nn.Linear(1024, 512)
            nn.init.kaiming_normal_(self.fc1.weight)
            self.fc2 = nn.Linear(512, 128)
            nn.init.kaiming_normal_(self.fc2.weight)
            self.fc3 = nn.Linear(130, num_outputs)
            nn.init.kaiming_normal_(self.fc3.weight)
            
    def forward(self, input_dict, state, seq_lens):
     
        features = input_dict["obs"]["features"].float()
        img = input_dict["obs"]["img"].float()/255
        x = self.pool(F.relu(self.conv1(img)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.cat([x, features], dim=1)
        x = F.relu(self.fc3(x))

        return x, []


class LowresCNN3D(TorchModelV2, nn.Module):
    ''' 3D CNN architecture using a sequence of occupancy grid map of size [n,10,70] and 2 additionals scalar features as input
    '''
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)
            
            self.conv1 = nn.Conv2d(6, 32, 3)
            nn.init.kaiming_normal_(self.conv1.weight)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3)
            nn.init.kaiming_normal_(self.conv2.weight)
            self.fc1 = nn.Linear(1024, 512)
            nn.init.kaiming_normal_(self.fc1.weight)
            self.fc2 = nn.Linear(512, 128)
            nn.init.kaiming_normal_(self.fc2.weight)
            self.fc3 = nn.Linear(130, num_outputs)
            nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, input_dict, state, seq_lens):
        features = input_dict["obs"][1].float()
        img = input_dict["obs"][0].float()/255
        x = self.pool(F.relu(self.conv1(img)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.cat([x, features], dim=1)
        x = self.fc3(x)
        return x, []


class LowresCNNLSTM(TorchModelV2, nn.Module):
    ''' CNN-LSTM architecture using a sequence of occupancy grid map of size [n,10,70] and 2 additionals scalar features as input
    '''
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)
            self.conv1 = nn.Conv2d(3, 32, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3)
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 128)
            self.lstm1 = nn.LSTM(128, 128, batch_first=True)
            self.fc3 = nn.Linear(130, num_outputs)
            
            nn.init.kaiming_normal_(self.conv1.weight)
            nn.init.kaiming_normal_(self.conv2.weight)
            nn.init.kaiming_normal_(self.fc1.weight)
            nn.init.kaiming_normal_(self.fc2.weight)
            nn.init.kaiming_normal_(self.fc3.weight)


    def forward(self, input_dict, state, seq_lens):
        features = input_dict["obs"][1].float()
        img_seq = input_dict["obs"][0].float()/255

        bs = img_seq.shape[0]
        hidden = None
        for t in range(img_seq.size(1)):
            x = self.pool(F.relu(self.conv1(img_seq[:, t, :, :, :])))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(bs, -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            out, hidden = self.lstm1(x.unsqueeze(0), hidden)  

        x = torch.cat([out[-1, :, :], features], dim=1)
        x = self.fc3(x)
        return x, []
            

            
            
            
class LowresCNNLSTM_test(TorchModelV2, nn.Module):
    ''' CNN-LSTM architecture using a sequence of occupancy grid map of size [n,10,70] and 2 additionals scalar features as input
        This architecture also contains a state predictive auxiliary task with its own loss function
    '''
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)


        self.conv_enc0 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv_enc1 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc_enc2 = nn.Sequential(nn.Linear(2176, 512), nn.ReLU())
        self.fc_enc3 =  nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.lstm1 = nn.LSTM(128, 128, batch_first=True)        
        self.fc3 = nn.Linear(130, num_outputs)
        
        self.tconv_dec0 = nn.Sequential(nn.ConvTranspose2d(32, 3, 2,stride=2), nn.Sigmoid())            
        self.tconv_dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3,stride=2), nn.ReLU())
        self.fc_dec2 = nn.Sequential(nn.Linear(512, 2176), nn.ReLU())
        self.fc_dec3 = nn.Sequential(nn.Linear(130, 512), nn.ReLU())

        self.view_requirements["future_obs"] = ViewRequirement(
            SampleBatch.OBS, shift=4, space=obs_space)
        self.features_value = None
        self.reset_params()
        
    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 

        
    def forward(self, input_dict, state, seq_lens):
        
        features = input_dict["obs"][1].float()
        img_seq = input_dict["obs"][0].float()/255
        bs = img_seq.shape[0]
        hidden = None

        for t in range(img_seq.size(1)):            

            self.hidden1 = self.conv_enc0(img_seq[:, t, :, :, :])
            self.hidden2 = self.conv_enc1(self.hidden1)
            self.hidden2_flat = self.hidden2.view(bs, -1)#2176
            self.hidden3 = self.fc_enc2(self.hidden2_flat)
            hidden4 = self.fc_enc3(self.hidden3)
            out, hidden = self.lstm1(hidden4.unsqueeze(0), hidden)  

        self.latent = torch.cat([out[-1, :, :], features], dim=1)
        
        
        #Value head
        self.features_value = self.latent
        x = self.fc3(self.features_value)

        return x, []         

    def custom_loss(self, policy_loss, loss_inputs):

        """Calculates a custom loss on top of the given policy_loss(es).
        Args:
            policy_loss (List[TensorType]): The list of already calculated
                policy losses (as many as there are optimizers).
            loss_inputs (TensorStruct): Struct of np.ndarrays holding the
                entire train batch.
        Returns:
            List[TensorType]: The altered list of policy losses. In case the
                custom loss should have its own optimizer, make sure the
                returned list is one larger than the incoming policy_loss list.
                In case you simply want to mix in the custom loss into the
                already calculated policy losses, return a list of altered
                policy losses (as done in this example below).
        """
        
        obs = restore_original_dimensions(
                  loss_inputs["future_obs"],
                  self.obs_space,
                  tensorlib="torch",
              )

        #State prediction head
        hidden5 = self.fc_dec3(self.latent)
        hidden6 = self.fc_dec2(hidden5)
        hidden6_2D = hidden6.view(self.hidden2.shape)
        hidden7 = self.tconv_dec1(hidden6_2D)
        out_aux = self.tconv_dec0(hidden7)
        
        target_aux = obs[0][:, -1, :, :,:].float()/255

        criterion = nn.BCELoss()
        loss_aux = criterion(out_aux, target_aux)

        self.auxiliary_loss_metric = loss_aux.item()
        self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])
        return [loss_ + 0.2 * loss_aux for loss_ in policy_loss]

    def metrics(self):
        return {
            "policy_loss": self.policy_loss_metric,
            "aux_loss": self.auxiliary_loss_metric,
        }            
    def value_function(self):
        return self.fc3(self.features_value)
             
    def future_state_prediction(self, obs):
        features = torch.Tensor(obs[1]).float().unsqueeze(0).cuda()
        img_seq = (torch.Tensor(obs[0]).float()/255).unsqueeze(0).cuda()
        bs = img_seq.shape[0]
        hidden = None

        for t in range(img_seq.size(1)):            


            self.hidden1 = self.conv_enc0(img_seq[:, t, :, :, :])
            self.hidden2 = self.conv_enc1(self.hidden1)
            self.hidden2_flat = self.hidden2.view(bs, -1)#2176
            self.hidden3 = self.fc_enc2(self.hidden2_flat)
            hidden4 = self.fc_enc3(self.hidden3)

            out, hidden = self.lstm1(hidden4.unsqueeze(0), hidden)  
            
        self.latent = torch.cat([out[-1, :, :], features], dim=1)
        
        hidden5 = self.fc_dec3(self.latent)
        hidden6 = self.fc_dec2(hidden5)
        hidden6_2D = hidden6.view(self.hidden2.shape)
        hidden7 = self.tconv_dec1(hidden6_2D)
        out_aux = self.tconv_dec0(hidden7)
        return out_aux



# Models need to be registered to be used in Ray and facilitate import
ModelCatalog.register_custom_model("LowresCNN", LowresCNN)
ModelCatalog.register_custom_model("LowresCNN3D", LowresCNN3D)
ModelCatalog.register_custom_model("LowresCNNLSTM", LowresCNNLSTM)
ModelCatalog.register_custom_model("LowresCNNLSTM_test", LowresCNNLSTM_test)

