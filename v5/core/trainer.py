"""
训练器（独立实现）- 支持 2D / 3D / VLA / VLA-Simple / RGB-JEPA
"""
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from data import DataStrategyRegistry
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logging import setup_logger


class Trainer:
    def __init__(self, cfg, strategy):
        self.cfg = cfg
        self.strategy = strategy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger()
        self.tokenizer = None

        self._load_data()
        self._build_model()
        self._setup_optimizer()

    def _load_data(self):
        dataset_cfg = OmegaConf.to_container(self.cfg.data.dataset, resolve=True)
        data_path = dataset_cfg.pop("name")

        model_type = getattr(self.cfg, 'model_type', '2d')

        if model_type in ['voxel', 'vla', 'vla_simple']:
            voxel_key = dataset_cfg.get('voxel_key', 'voxel')
            self.dataset = self.strategy.load(
                data_path,
                keys_to_load=dataset_cfg.get("keys_to_load", ['voxel', 'action']),
                transform=self.strategy.get_transform(self.cfg.voxel_size),
                num_steps=dataset_cfg.get('num_steps', 4),
                voxel_key=voxel_key
            )
            sample = self.dataset[0]
            action_dim = sample["action"].shape[-1]
            if hasattr(self.cfg.model, "action_encoder"):
                self.cfg.model.action_encoder.input_dim = action_dim
            print(f"[Train] Action dimension: {action_dim}")
        else:
            self.dataset = self.strategy.load(
                data_path,
                keys_to_load=dataset_cfg.get("keys_to_load", ['pixels', 'action']),
                transform=self.strategy.get_transform(self.cfg.img_size),
                num_steps=dataset_cfg.get('num_steps', 4)
            )
            sample = self.dataset[0]
            action_dim = sample["action"].shape[-1]
            if hasattr(self.cfg.model, "action_encoder"):
                self.cfg.model.action_encoder.input_dim = action_dim
            print(f"[Train] Action dimension: {action_dim}")

        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_set, self.val_set = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.cfg.loader.batch_size,
            shuffle=True,
            num_workers=self.cfg.loader.num_workers,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.cfg.loader.batch_size,
            shuffle=False,
            num_workers=self.cfg.loader.num_workers,
        )
        self.logger.info(f"Train samples: {len(self.train_set)}, Val samples: {len(self.val_set)}")

    def _build_model(self):
        import hydra
        model_cfg = OmegaConf.to_container(self.cfg.model, resolve=True)

        model_type = getattr(self.cfg, 'model_type', '2d')

        if model_type == 'voxel':
            from core.voxel import VoxelJEPAEncoder
            from core.model_3d import JEPA3D

            voxel_spec = getattr(self.cfg, 'voxel_spec', None)
            if voxel_spec is None:
                from core.voxel import VoxelSpec
                voxel_spec = VoxelSpec()

            embed_dim = self.cfg.embed_dim

            encoder = VoxelJEPAEncoder(spec=voxel_spec, latent_dim=embed_dim)
            predictor = hydra.utils.instantiate(model_cfg["predictor"])
            action_encoder = hydra.utils.instantiate(model_cfg["action_encoder"])
            projector = hydra.utils.instantiate(model_cfg["projector"])
            pred_proj = hydra.utils.instantiate(model_cfg["pred_proj"])

            self.model = JEPA3D(
                encoder=encoder,
                predictor=predictor,
                action_encoder=action_encoder,
                projector=projector,
                pred_proj=pred_proj,
            )

        elif model_type == 'vla':
            from core.voxel import VoxelJEPAEncoder
            from core.vla_jepa import VLAJEPA

            voxel_spec = getattr(self.cfg, 'voxel_spec', None)
            if voxel_spec is None:
                from core.voxel import VoxelSpec
                voxel_spec = VoxelSpec()

            embed_dim = self.cfg.embed_dim

            encoder = VoxelJEPAEncoder(spec=voxel_spec, latent_dim=embed_dim)
            predictor = hydra.utils.instantiate(model_cfg["jepa_predictor"])
            action_encoder = hydra.utils.instantiate(model_cfg["action_encoder"])
            projector = hydra.utils.instantiate(model_cfg["projector"])
            pred_proj = hydra.utils.instantiate(model_cfg["pred_proj"])

            self.model = VLAJEPA(
                jepa_encoder=encoder,
                jepa_predictor=predictor,
                action_encoder=action_encoder,
                projector=projector,
                pred_proj=pred_proj,
                llm_model_name=model_cfg.get("llm_model_name", "google/bert_uncased_L-2_H-128_A-2"),
                llm_dim=model_cfg.get("llm_dim", 128),
                jepa_dim=embed_dim,
                num_heads=model_cfg.get("num_heads", 4),
                use_adaln=model_cfg.get("use_adaln", False),
                freeze_llm=model_cfg.get("freeze_llm", True),
            )

        elif model_type == 'vla_simple':
            from core.voxel import VoxelJEPAEncoder
            from core.vla_simple import VLASimple

            voxel_spec = getattr(self.cfg, 'voxel_spec', None)
            if voxel_spec is None:
                from core.voxel import VoxelSpec
                voxel_spec = VoxelSpec()

            embed_dim = self.cfg.embed_dim

            encoder = VoxelJEPAEncoder(spec=voxel_spec, latent_dim=embed_dim)
            predictor = hydra.utils.instantiate(model_cfg["jepa_predictor"])
            action_encoder = hydra.utils.instantiate(model_cfg["action_encoder"])
            projector = hydra.utils.instantiate(model_cfg["projector"])
            pred_proj = hydra.utils.instantiate(model_cfg["pred_proj"])

            self.model = VLASimple(
                jepa_encoder=encoder,
                jepa_predictor=predictor,
                action_encoder=action_encoder,
                projector=projector,
                pred_proj=pred_proj,
                latent_dim=embed_dim,
                num_actions=8,
            )

        elif model_type == 'rgb_jepa':
            from core.rgb_jepa import RGBDJEPA

            self.model = RGBDJEPA(
                img_size=self.cfg.img_size,
                in_chans=self.strategy.get_input_channels(),
                embed_dim=192,
                dropout=0.1,
            )

        elif model_type == 'rgb_action':
            from core.rgb_action import RGBDActionModel

            self.model = RGBDActionModel(
                img_size=self.cfg.img_size,
                in_chans=self.strategy.get_input_channels(),
                num_actions=8,
                dropout=0.3,
            )

        elif model_type == 'rgb_vla':
            from core.rgb_vla import RGBVLA

            self.model = RGBVLA(
                img_size=self.cfg.img_size,
                in_chans=self.strategy.get_input_channels(),
                num_actions=8,
                dropout=0.3,
                llm_model_name=model_cfg.get("llm_model_name", "google/bert_uncased_L-2_H-128_A-2"),
                llm_dim=model_cfg.get("llm_dim", 128),
                num_heads=model_cfg.get("num_heads", 4),
                freeze_llm=model_cfg.get("freeze_llm", True),
            )

        else:
            # 2D 模型
            model_cfg["encoder"]["in_chans"] = self.strategy.get_input_channels()
            self.model = hydra.utils.instantiate(model_cfg)

        self.model = self.model.to(self.device)

        from legacy.module import SIGReg
        self.sigreg = SIGReg(**self.cfg.loss.sigreg.kwargs).to(self.device)

        self.logger.info(f"Model built with {sum(p.numel() for p in self.model.parameters())} params")

    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay
        )

    def run(self):
        self.logger.info("Starting training...")
        for epoch in range(self.cfg.trainer.max_epochs):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)

            if (epoch + 1) % 1 == 0:
                save_checkpoint(
                    self.model,
                    self.cfg.output_model_name,
                    epoch + 1,
                    self.cfg
                )
        self.logger.info("Training complete!")

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            output = self.model(batch)

            if "losses" in output:
                loss = output["losses"]["total_loss"]
                pred_loss = output["losses"].get("pred_loss", torch.tensor(0.0))
            elif "loss" in output:
                loss = output["loss"]
                pred_loss = output.get("pred_loss", torch.tensor(0.0))
            else:
                loss = output.get("loss", torch.tensor(0.0))
                pred_loss = output.get("pred_loss", torch.tensor(0.0))

            if hasattr(self, 'sigreg') and "emb" in output:
                sigreg_loss = self.sigreg(output["emb"].transpose(0, 1))
                loss = loss + self.cfg.loss.sigreg.weight * sigreg_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "pred": pred_loss.item() if torch.is_tensor(pred_loss) else pred_loss})

        avg_loss = total_loss / len(self.train_loader)
        self.logger.info(f"Epoch {epoch+1} train loss: {avg_loss:.4f}")

    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}
                output = self.model(batch)

                if "losses" in output:
                    loss = output["losses"]["total_loss"]
                elif "loss" in output:
                    loss = output["loss"]
                else:
                    loss = output.get("loss", torch.tensor(0.0))

                if hasattr(self, 'sigreg') and "emb" in output:
                    sigreg_loss = self.sigreg(output["emb"].transpose(0, 1))
                    loss = loss + self.cfg.loss.sigreg.weight * sigreg_loss

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.logger.info(f"Epoch {epoch+1} val loss: {avg_loss:.4f}")
