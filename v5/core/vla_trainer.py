"""
VLA-JEPA训练器
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from .vla_jepa import VLAJEPA
from utils.checkpoint import save_checkpoint
from utils.logging import setup_logger


class VLATrainer:
    def __init__(self, cfg, strategy):
        self.cfg = cfg
        self.strategy = strategy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger()
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._load_data()
        self._build_model()
        self._setup_optimizer()
    
    def _load_data(self):
        dataset_cfg = OmegaConf.to_container(self.cfg.data.dataset, resolve=True)
        data_path = dataset_cfg.pop("name")
        
        self.dataset = self.strategy.load(
            data_path,
            keys_to_load=dataset_cfg.get("keys_to_load", ['voxel', 'action']),
            transform=self.strategy.get_transform(self.cfg.voxel_spec),
            num_steps=dataset_cfg.get('num_steps', 4),
            tokenizer=self.tokenizer,
            max_length=self.cfg.llm.max_length,
        )
        
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
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.cfg.loader.batch_size,
            shuffle=False,
            num_workers=self.cfg.loader.num_workers,
        )
        self.logger.info(f"Train samples: {len(self.train_set)}, Val samples: {len(self.val_set)}")
    
    def _build_model(self):
        # 创建模型
        model_cfg = OmegaConf.to_container(self.cfg.model, resolve=True)
        import hydra
        self.model = hydra.utils.instantiate(model_cfg)
        self.model = self.model.to(self.device)
        
        self.logger.info(f"Model built with {sum(p.numel() for p in self.model.parameters())} params")
    
    def _setup_optimizer(self):
        # 只训练可训练参数（LLM被冻结）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
    
    def run(self):
        self.logger.info("Starting VLA-JEPA training...")
        for epoch in range(self.cfg.trainer.max_epochs):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            
            if (epoch + 1) % 1 == 0:
                save_checkpoint(
                    self.model,
                    self.cfg.output_model_name,
                    epoch + 1,
                    self.cfg,
                )
        self.logger.info("Training complete!")
    
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            # 数据移到GPU
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # 前向传播
            output = self.model(batch)
            loss = output['losses']['total_loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        self.logger.info(f"Epoch {epoch+1} train loss: {avg_loss:.4f}")
    
    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                output = self.model(batch)
                total_loss += output['losses']['total_loss'].item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.logger.info(f"Epoch {epoch+1} val loss: {avg_loss:.4f}")