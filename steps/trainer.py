import time
import os, random
import torch
import math, pickle
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import logging
from data import gigaspeech
from models import voicecraft

from .trainer_utils import DistributedDynamicBatchSampler, StatefulDistributedSampler, AverageMeter, print_model_info
from .optim import ScaledAdam, Eden


class Trainer:
    
    def __init__(self, args, world_size, rank):
        self.start_time = time.time()
        self.args = args
        self.world_size, self.rank = world_size, rank
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        if self.rank == 0:
            self.writer = SummaryWriter(args.exp_dir)
        self.seed_everything(seed=self.args.seed)
        self.meters = self._setup_meters()

        self.progress, self.total_progress = self._setup_progress()

        self.model, self.trainables, self.optim_states, self.scheduler_states = self._setup_models()

        self.train_dataset_length, self.train_sampler, self.train_loader, self.valid_loader = self._setup_dataloader()
        if self.args.num_steps != None:
            self.total_step = self.args.num_steps
            self.args.num_epochs = math.ceil(self.total_step / math.floor(self.train_dataset_length / self.args.batch_size)) if not self.args.dynamic_batching else None
        else:
            self.total_step = int(math.floor(self.train_dataset_length / self.args.batch_size))*self.args.num_epochs

        self.optimizer, self.scheduler = self._setup_optimizer()
        self.scaler = torch.cuda.amp.GradScaler()
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank], find_unused_parameters=False)
        
        if self.rank == 0:
            self.early_stop_accu_steps = 0
            if self.args.dynamic_batching:
                logging.info(f"max number of tokens per GPU in a training batch: {self.args.max_num_tokens}, max number of tokens per GPU in a inference batch: {self.args.val_max_num_tokens}")
            else:
                logging.info(f"batch size (summed over all GPUs): {self.args.batch_size}")

    def train(self):
        flag = True
        skip_flag = False
        data_start_time = time.time()
        while flag:
            self.train_sampler.set_epoch(self.progress['epoch'])
            for i, batch in enumerate(self.train_loader):
                data_end_time = time.time()
                self.model.train()
                if self.progress['step'] > self.total_step:
                    flag = False
                    self.validate_and_save()
                    if self.rank == 0:
                        self.writer.close()
                    break
                if isinstance(self.scheduler, Eden):
                    self.scheduler.step_epoch(self.progress['step']//self.args.pseudo_epoch_size + 1)
                if self.args.optimizer_name == "ScaledAdam":
                    cur_lr = self.scheduler.get_last_lr()[0]
                else:
                    lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
                    assert lrs[0] == lrs[1]
                    cur_lr = lrs[0]

                if self.rank == 0 and self.progress['step'] % self.args.tb_write_every_n_steps == 0:
                    self.writer.add_scalar("train/lr", cur_lr, self.progress['step'])

                all_inds = list(range(len(batch['y'])))
                sum_losses = 0
                sum_top10acc = 0
                sum_ntoken = 0
                sum_top10acc_cbi = [0 for _ in range(self.args.n_codebooks)]
                for j in range(self.args.gradient_accumulation_steps):
                    cur_ind = all_inds[j::self.args.gradient_accumulation_steps]
                    cur_batch = {key: batch[key][cur_ind] for key in batch}
                    with torch.cuda.amp.autocast(dtype=torch.float16 if self.args.precision=="float16" else torch.float32):
                        out = self.model(cur_batch)
                        if out == None:
                            continue

                    record_loss = out['loss'].detach().to(self.rank) 
                    top10acc = out['top10acc'].to(self.rank)
                    effective_ntoken = out['effective_ntoken'].to(self.rank)
                    is_nan = torch.tensor(int(torch.isnan(record_loss).any()), dtype=torch.float32, device=self.rank)
                    
                    dist.all_reduce(record_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(top10acc, op=dist.ReduceOp.SUM)
                    dist.all_reduce(effective_ntoken, op=dist.ReduceOp.SUM)
                    dist.all_reduce(is_nan, op=dist.ReduceOp.SUM)
                    
                    # check if loss is nan
                    if is_nan.item() > 0:
                        logging.info(f"loss at step {self.progress['step']} is nan, therefore skip this batch")
                        skip_flag = True
                        continue

                    sum_losses += record_loss.item()
                    sum_top10acc += top10acc.item()
                    sum_ntoken += effective_ntoken.item()

                    if 'top10acc_by_codebook' in out:
                        for cb in range(self.args.n_codebooks):
                            top10acc_cbi = out['top10acc_by_codebook'][cb]
                            dist.all_reduce(top10acc_cbi, op=dist.ReduceOp.SUM)
                            sum_top10acc_cbi[cb] += top10acc_cbi.item()
                        
                    if self.rank == 0:
                        average_loss = sum_losses / sum_ntoken
                        average_top10acc = sum_top10acc / sum_ntoken
                        self.meters['train_loss'].update(average_loss, batch['x'].shape[0]*self.world_size)
                        self.meters['train_top10acc'].update(average_top10acc, batch['x'].shape[0]*self.world_size)
                        self.meters['train_top10acc'].update(average_top10acc, batch['x'].shape[0]*self.world_size)
                        average_top10acc_cbi = [sum_top10acc_cbi[cb] / sum_ntoken * self.args.n_codebooks for cb in range(self.args.n_codebooks)]
                        for cb in range(self.args.n_codebooks):
                            self.meters[f'train_top10acc_cb{cb+1}'].update(average_top10acc_cbi[cb], batch['x'].shape[0]*self.world_size)

                        if self.progress['step'] % self.args.tb_write_every_n_steps == 0:
                            self.writer.add_scalar('train/loss', average_loss, self.progress['step'])
                            self.writer.add_scalar('train/top10acc', average_top10acc, self.progress['step'])
                            self.writer.add_scalar("train/ntokens", sum_ntoken, self.progress['step'])
                            for cb in range(self.args.n_codebooks):
                                self.writer.add_scalar(f'train/top10acc_cb{cb+1}', average_top10acc_cbi[cb], self.progress['step'])

                    if self.args.optimizer_name == "ScaledAdam":
                        self.scaler.scale(out['loss']).backward() 
                    else:
                        self.scaler.scale(out['loss']/out['effective_ntoken']).backward()

                if skip_flag:
                    self.optimizer.zero_grad()
                    skip_flag = False
                    continue

                if self.args.optimizer_name != "ScaledAdam":
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad()

                if self.args.optimizer_name == "ScaledAdam":
                    self.scheduler.step_batch(self.progress['step'])
                else:
                    self.scheduler.step()

                if self.rank == 0:
                    self.meters['data_time'].update(data_end_time - data_start_time)
                    self.meters['train_time'].update(time.time() - data_end_time)
                    if self.progress['step'] % self.args.tb_write_every_n_steps == 0:
                        self.writer.add_scalar("train/data_time", data_end_time - data_start_time, self.progress['step'])
                        self.writer.add_scalar("train/train_time", time.time() - data_end_time, self.progress['step'])
                        

                    # logging
                    if self.progress['step'] % self.args.print_every_n_steps == 0:
                        log_out = {}
                        log_out['cur_epoch'] = f"{self.progress['epoch']}/{self.args.num_epochs}" if self.args.num_epochs is not None else f"{self.progress['epoch']}"
                        log_out['cur_step'] = f"{int(self.progress['cur_step']+1)}"
                        log_out['total_step'] = f"{self.progress['step']}/{self.args.num_steps}"
                        log_out['lr'] = f"{cur_lr:.7f}"
                        log_out['ntokens'] = f"{sum_ntoken}"
                        for key in self.meters:
                            if self.meters[key].val != 0 or self.meters[key].avg != 0:
                                log_out[key] = f"{self.meters[key].val:.4f} ({self.meters[key].avg:.4f})" if isinstance(self.meters[key].val, float) else f"{self.meters[key].val}"
                        logging.info(log_out)
                        if np.isnan(self.meters['train_loss'].avg):
                            logging.warning("training diverged...")
                            raise RuntimeError("training diverged...")

                # validation and save models
                if self.progress['step'] % self.args.val_every_n_steps == 0:
                    dist.barrier()
                    self.validate_and_save()

                self.progress['step'] += 1
                self.progress['cur_step'] += 1

                data_start_time = time.time()
            self.progress['epoch'] += 1
            self.progress['cur_step'] = 0 # reset cur_step to be 0
        dist.destroy_process_group()

    def validate_and_save(self):
        self.model.eval()
        
        score = self.validate(self.valid_loader)

        if self.rank == 0:
            if self.args.early_stop_threshold > 0:
                if self.progress['best_score'] - score < self.args.early_stop_threshold:
                    self.early_stop_accu_steps += self.args.val_every_n_steps
                    if self.early_stop_accu_steps >= self.args.early_stop_step-1:
                        logging.info(f"early stop based on self.args.early_stop_threshold: {self.args.early_stop_threshold}, and self.args.early_stop_step: {self.args.early_stop_step}")
                        logging.info(f"best validation score at step: {self.progress['best_step']}, and the score is {self.progress['best_score']:.4f}")
                        dist.destroy_process_group()
                        raise RuntimeError("early stop")
                else:
                    self.early_stop_accu_steps = 0

            if (score < self.progress['best_score']):
                self.progress['best_step'] = self.progress['step']
                self.progress['best_score'] = score
                save_path = os.path.join(self.args.exp_dir,"best_bundle.pth")
                torch.save(
                    {
                        "model": self.model.module.state_dict(),
                        "optimizer":  self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "config": self.args,
                        "phn2num": self.train_loader.dataset.phn2num
                    },save_path
                )
                logging.info(f"save *best* models at {save_path} at global step {self.progress['step']}")
            self._save_progress()
            save_path = os.path.join(self.args.exp_dir,"bundle.pth")
            torch.save(
                {
                    "model": self.model.module.state_dict(),
                    "optimizer":  self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "config": self.args,
                    "phn2num": self.train_loader.dataset.phn2num
                    },save_path
            )
            logging.info(f"save models, indices, acc and other statistics at {save_path} and {self.args.exp_dir}/progress.pkl at global step {self.progress['step']}")

        dist.barrier()

    def validate(self, valid_loader=None, hide_progress=True):
        if valid_loader == None:
            valid_loader = self.valid_loader
        self.model.eval()

        start_val_time = time.time()
        sum_losses = 0
        sum_top10acc = 0
        sum_ntoken = 0
        sum_top10acc_cbi = [0 for _ in range(self.args.n_codebooks)]

        with torch.no_grad():
            for i, batch in enumerate(tqdm(valid_loader, disable=hide_progress)):
                out = self.model(batch)
                sum_losses += out['loss']
                sum_top10acc += out['top10acc']
                sum_ntoken += out['effective_ntoken']
                if 'top10acc_by_codebook' in out:
                    for cb in range(self.args.n_codebooks):
                        sum_top10acc_cbi[cb] += out['top10acc_by_codebook'][cb]
                        
        dist.all_reduce(sum_losses, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_top10acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_ntoken, op=dist.ReduceOp.SUM)
        
        if 'top10acc_by_codebook' in out:
            for cb in range(self.args.n_codebooks):
                dist.all_reduce(sum_top10acc_cbi[cb], op=dist.ReduceOp.SUM)
        
        if self.rank == 0:
            val_loss = sum_losses / sum_ntoken
            val_top10acc = sum_top10acc / sum_ntoken
            # logging
            self.meters['val_loss'].update(val_loss)
            logging.info(f"val loss: {val_loss:.5f}")
            self.writer.add_scalar("val/loss", val_loss, self.progress['step'])

            self.meters['val_top10acc'].update(val_top10acc)
            logging.info(f"val top10acc: {val_top10acc:.5f}")
            self.writer.add_scalar("val/top10acc", val_top10acc, self.progress['step'])
            for cb in range(self.args.n_codebooks):
                average_top10acc_cbi = sum_top10acc_cbi[cb] / sum_ntoken * self.args.n_codebooks
                self.meters[f'val_top10acc_cb{cb+1}'].update(average_top10acc_cbi)
                self.writer.add_scalar(f'val/top10acc_cb{cb+1}', average_top10acc_cbi, self.progress['step'])

            logging.info(f"validation takes: {time.time() - start_val_time:.2f}s")
            logging.info(f"Step [{self.progress['step']}/{self.total_step}]\t Time elapsed {(time.time() - self.start_time)/3600.:.2f}h, Val Loss: {val_loss:.4f}, Val Top10Acc: {val_top10acc:.4f}")
            return val_loss.item()
        else:
            return None

    def _setup_meters(self):
        meters = {}
        meter_names = ['train_loss', 'val_loss', 'train_top10acc', 'val_top10acc', 'data_time', 'train_time']
        meter_names += ['train_dur_loss', 'train_dur_acc', 'val_dur_loss', 'val_dur_acc']
        meter_names += [f'train_top10acc_cb{cb+1}' for cb in range(self.args.n_codebooks)]
        meter_names += [f'val_top10acc_cb{cb+1}' for cb in range(self.args.n_codebooks)]
        for name in meter_names:
            meters[name] = AverageMeter()
        return meters
    def _setup_progress(self):
        progress = {}
        progress['best_step'] = 1
        progress['best_score'] = np.inf # this records loss value
        progress['step'] = 1
        progress['epoch'] = 1
        progress['cur_step'] = 0 # step in the current epoch, for resuming the sampler
        total_progress = []
        # if self.args.resume or self.args.validate:
        if self.args.resume:
            progress_pkl = "%s/progress.pkl" % self.args.exp_dir
            with open(progress_pkl, "rb") as f:
                total_progress = pickle.load(f)
                progress['best_step'], progress['best_score'], progress['step'], progress['epoch'], progress['cur_step'], _ = total_progress[-1]
            if self.rank == 0:
                logging.info("\nResume training from:")
                logging.info("  epoch = %s" % progress['epoch'])
                logging.info("  cur_step = %s" % progress['cur_step'])
                logging.info("  step = %s" % progress['step'])
                logging.info("  best_step = %s" % progress['best_step'])
                logging.info("  best_score = %s" % progress['best_score'])
        return progress, total_progress
    
    def _save_progress(self):
        self.total_progress.append([self.progress['best_step'], self.progress['best_score'], int(self.progress['step']+1), self.progress['epoch'], int(self.progress['cur_step']+1), time.time() - self.start_time])
        with open("%s/progress.pkl" % self.args.exp_dir, "wb") as f:
            pickle.dump(self.total_progress, f)

    def _setup_dataloader(self):
        assert self.args.dataset == 'gigaspeech', "only gigaspeech is supported for now"
        train_dataset, val_dataset = gigaspeech.dataset(self.args, 'train'), gigaspeech.dataset(self.args, 'validation')
        
        if self.args.dynamic_batching:
            train_sampler = DistributedDynamicBatchSampler(train_dataset, self.args, num_replicas=self.world_size, rank=self.rank, shuffle=True, seed=self.args.seed, drop_last=True, lengths_list=train_dataset.lengths_list, verbose=True, epoch=0)
            valid_sampler = DistributedDynamicBatchSampler(val_dataset, self.args, num_replicas=self.world_size, rank=self.rank, shuffle=True, seed=self.args.seed, drop_last=True, lengths_list=val_dataset.lengths_list, verbose=True, epoch=0)
        else:
            train_sampler = StatefulDistributedSampler(train_dataset, self.args.batch_size//self.world_size, num_replicas=self.world_size, rank=self.rank, shuffle=True, seed=self.args.seed, drop_last=True)
            valid_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False, seed=self.args.seed, drop_last=False)
            
        if self.progress['step'] > 1:
            train_sampler.set_epoch_resume(self.progress['epoch'], self.progress['cur_step'])

        if self.args.dynamic_batching:
            train_loader = torch.utils.data.DataLoader(train_dataset, 
                            batch_sampler=train_sampler, 
                            num_workers=self.args.num_workers//self.world_size,
                            collate_fn=train_dataset.collate, persistent_workers=True
                            )
            valid_loader = torch.utils.data.DataLoader(val_dataset, 
                            batch_sampler=valid_sampler, 
                            num_workers=self.args.num_workers//self.world_size,
                            collate_fn=val_dataset.collate, persistent_workers=True
                            )
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, 
                            batch_size=self.args.batch_size//self.world_size, sampler=train_sampler, num_workers=self.args.num_workers//self.world_size,
                            collate_fn=train_dataset.collate, persistent_workers=True
                            )
            valid_loader = torch.utils.data.DataLoader(val_dataset, 
                            batch_size=self.args.batch_size//self.world_size, sampler=valid_sampler,
                            num_workers=self.args.num_workers//self.world_size,
                            collate_fn=val_dataset.collate, persistent_workers=True
                            )
        return len(train_dataset), train_sampler, train_loader, valid_loader
        

        
    def _setup_models(self):
        model = voicecraft.VoiceCraft(self.args)

        if self.rank == 0:
            logging.info(model)
            logging.info("model parameters")
            print_model_info(model)

        if self.progress['step'] > 1:
            bundle = torch.load(os.path.join(self.args.exp_dir, "bundle.pth"), map_location="cpu")
            model.load_state_dict(bundle['model'])
            optim_states = bundle['optimizer']
            scheduler_states = bundle['scheduler']
            if self.rank == 0:
                logging.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['step']))
            del bundle['model']
        else:
            optim_states = None
            scheduler_states = None

        if self.args.load_model_from != None and self.progress['step'] <= 1:
            sd = torch.load(self.args.load_model_from, map_location="cpu")['model']
            model.load_state_dict(sd)
            del sd
        
        if self.args.optimizer_name == "ScaledAdam":
            trainables = [p for p in model.parameters() if p.requires_grad]
        else:
            no_decay = [".bias", ".audio_embeddings.weight", ".text_embeddings.weight", ".norm.weight", ".norm1.weight", ".norm2.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]
            if len(optimizer_grouped_parameters[1]['params']) == 0:
                logging.info("there is no embedding weights, bias, and layernorm parameters in the model, which should be True, check model parameter names")
                trainables = optimizer_grouped_parameters[0]
            else:
                trainables = optimizer_grouped_parameters
        model.to(self.device)

        return model, trainables, optim_states, scheduler_states

    
    def _setup_optimizer(self):
        if self.args.optimizer_name == "ScaledAdam":
            parameters_names = []
            parameters_names.append([n for n,p in self.model.named_parameters() if p.requires_grad])
            optimizer = ScaledAdam(
                self.trainables,
                lr=self.args.lr,
                betas=(0.9, 0.95),
                clipping_scale=2.0,
                parameters_names=parameters_names,
                show_dominant_parameters=False,
                clipping_update_period=self.args.clipping_update_period,
            )
            scheduler = Eden(optimizer, self.args.reduce_lr_start_step, self.args.reduce_lr_start_epoch, warmup_batches=self.total_step * self.args.warmup_fraction)

        else:
            optimizer = AdamW(self.trainables, lr=self.args.lr)
            warmup_steps = self.total_step * self.args.warmup_fraction
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    0.0, float(self.total_step - current_step) / float(max(1, self.total_step - warmup_steps))
                )

            scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
            
        # if resume
        if self.progress['step'] > 1:
            optimizer.load_state_dict(self.optim_states)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            del self.optim_states

            scheduler.load_state_dict(self.scheduler_states)

        optimizer.zero_grad()
        return optimizer, scheduler
    
    def seed_everything(self, seed=1):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True