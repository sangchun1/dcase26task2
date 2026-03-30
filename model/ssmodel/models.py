from dataclasses import dataclass
import fairseq
import torch.nn as nn
import torch
from model.ssmodel.unilm.beats_lora.BEATs import BEATs, BEATsConfig

import loralib as lora

@dataclass
class UserDirModule: 
    user_dir: str

class audio_feature_extractor(nn.Module):
    def __init__(self, model_name, use_lora, aggregation=False, adaptor_lora=False):
        super().__init__()
        if model_name=="eat":
            if use_lora:
                model_path = UserDirModule('/home/user/KMJ/work/ASD/2026/model/ssmodel/fairseq/EAT_lora')
            else:
                model_path = UserDirModule('/home/user/KMJ/work/ASD/2026/model/ssmodel/fairseq/EAT')
            fairseq.utils.import_user_module(model_path)
            checkpoint_dir = '/home/user/hyunjun/pretrained_weights/EAT/EAT-base_epoch30_pt.pt'
            
            if use_lora:
                model, config, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir], strict=False)
            else:
                model, config, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
            
            self.model = model[0]
        
        elif model_name=='beats':
            
            model_path = '/home/user/hyunjun/pretrained_weights/BEATs_iter3_plus_AS2M.pt'
            checkpoint = torch.load(model_path)
            cfg = BEATsConfig(checkpoint['cfg'])
            self.model = BEATs(cfg).float()
            if use_lora:
                self.model.load_state_dict(checkpoint['model'], strict=False)
            else:
                self.model.load_state_dict(checkpoint['model'])
            
            # LayerNorm을 FP32로 고정
            for module in self.model.modules():
                if isinstance(module, nn.LayerNorm):
                    module = module.to(dtype=torch.float32)

        if aggregation:
            adaptor = []
            in_dim = int(12*768)
            adaptor.append(nn.LayerNorm(in_dim))
            adaptor.append(lora.Linear(in_dim, in_dim, r=64) if adaptor_lora else nn.Linear(in_dim, in_dim))
            adaptor.append(nn.GELU())
            adaptor.append(lora.Linear(in_dim, 768, r=64) if adaptor_lora else nn.Linear(in_dim, 768))
            self.adaptor = nn.Sequential(*adaptor)

        self.aggregation = aggregation
        self.model_name = model_name

    def forward(self, x, padding_mask, spec_aug=False):
        if self.model_name=='eat':
            if self.aggregation:
                def hook_fn(module, input, output, layer_outputs):
                    x = output[0]
                    if x.shape[1] == 513:
                        x = x[:, 1:, :] # remove class token
                    layer_outputs.append(x)

                layer_outputs = []
                hooks = []
                for block in self.model.blocks:
                    hook = block.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, layer_outputs))
                    hooks.append(hook)

                _ = self.model.extract_features(x, padding_mask=None, mask=False, remove_extra_tokens=False)

                for hook in hooks:
                    hook.remove()

                concatenated = torch.cat(layer_outputs, dim=-1)
                # print("Before adaptor: ", concatenated.shape)
                feats = self.adaptor(concatenated)
                # print("After adaptor: ", feats.shape)
                # print("aggregation is True")

            else:
                feats = self.model.extract_features(x, padding_mask=None, mask=False, remove_extra_tokens=True)
                feats = feats['x']
                # print("aggregation is False")
        
        elif self.model_name=='beats':
            # CustomDataset 은 BEATs 용 source 를 [B, 1, T] 로 stack 함.
            # BEATs.preprocess 는 source 가 [B, T] 일 때 각 행이 [T] 가 되고,
            # 내부 unsqueeze(0) 후 [1, T] 로 fbank(c, n) 형식이 됨.
            # [B, 1, T] 그대로 넘기면 [1, T] 에 unsqueeze 가 겹쳐 [1, 1, T] 가 되어 torchaudio fbank 가 실패함.
            if x.dim() == 3 and x.size(1) == 1:
                x = x.squeeze(1)

            if self.aggregation:
                layer_outputs = self.model.extract_features(source=x, padding_mask=padding_mask, spec_aug=spec_aug)[-1]
                concatenated = torch.cat([o[0].transpose(0,1) for o in layer_outputs], dim=-1)
                feats = self.adaptor(concatenated)

            else:
                feats = self.model.extract_features(source=x, padding_mask=padding_mask, spec_aug=spec_aug)[0]

        return feats