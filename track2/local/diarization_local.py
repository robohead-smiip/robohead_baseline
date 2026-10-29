"""Feed causal acoustic features into NeMo streaming diarization."""

from copy import deepcopy
from types import SimpleNamespace

import torch

from robohead_baseline.track2.local.features import IncrementalFrontend
from robohead_baseline.utils.assets import nemo_checkpoint


class NeMoDiarization:
    def __init__(self, config):
        from omegaconf import OmegaConf
        from nemo.collections.asr.models import SortformerEncLabelModel
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import configure_diar_streaming

        self.model = SortformerEncLabelModel.restore_from(
            nemo_checkpoint(config['diarization_model']), map_location='cuda:0'
        )
        self.bf16 = torch.cuda.is_bf16_supported()
        self.model = self.model.to(dtype=torch.bfloat16 if self.bf16 else torch.float32).eval()
        settings = config['streaming']
        settings = SimpleNamespace(
            streaming_mode=True,
            diar_right_context=0,
            spkcache_len=settings['spkcache_len'],
            fifo_len=settings['fifo_len'],
            spkcache_update_period=settings['spkcache_update_period'],
            log=False,
        )
        configure_diar_streaming(
            self.model, settings, output_subsampling_factor=8, diar_chunk_len=config['streaming']['sd_chunk_frames'],
        )
        cfg = deepcopy(self.model._cfg.preprocessor)
        OmegaConf.set_struct(cfg, False)
        cfg.dither, cfg.pad_to = 0.0, 0
        self.preprocessor = self.model.from_config_dict(cfg).to('cuda:0').eval()
        self.chunk_frames = config['streaming']['sd_chunk_frames'] * 8

    def reset(self):
        self.frontend = IncrementalFrontend(self)
        self.features = None
        self.state = self.model.sortformer_modules.init_streaming_state(
            batch_size=1, async_streaming=self.model.async_streaming, device=self.model.device
        )
        self.predictions = torch.zeros((1, 0, self.model.sortformer_modules.n_spk), device=self.model.device)

    def preprocess_audio(self, audio):
        signal = torch.as_tensor(audio, device=self.model.device).unsqueeze(0)
        length = torch.tensor([len(audio)], device=self.model.device)
        return self.preprocessor(input_signal=signal, length=length)

    def push(self, pcm, eof=False):
        features = self.frontend.accept(pcm, eof)
        if features is not None:
            self.features = features if self.features is None else torch.cat((self.features, features), dim=-1)
        first = self.predictions.shape[1]
        while self.features is not None and (self.features.shape[-1] >= self.chunk_frames or eof):
            if not self.features.shape[-1]:
                break
            chunk = self.features[:, :, : self.chunk_frames]
            self.features = self.features[:, :, self.chunk_frames :]
            self.state, self.predictions = self.model.forward_streaming_step(
                processed_signal=chunk.transpose(1, 2),
                processed_signal_length=torch.tensor([chunk.shape[-1]], device=self.model.device),
                streaming_state=self.state,
                total_preds=self.predictions,
                drop_extra_pre_encoded=0,
                right_offset=0,
            )
        return self.predictions[0, first:].float().cpu().numpy()
