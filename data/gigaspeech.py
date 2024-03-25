import os
import torch
import random
import copy
import logging
import shutil

class dataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.split = split
        assert self.split in ['train', 'validation', 'test']
        manifest_fn = os.path.join(self.args.dataset_dir, self.args.manifest_name, self.split+".txt")

        with open(manifest_fn, "r") as rf:
            data = [l.strip().split("\t") for l in rf.readlines()]
        lengths_list = [int(item[-1]) for item in data]
        self.data = []
        self.lengths_list = []
        for d, l in zip(data, lengths_list):
            if l >= self.args.encodec_sr*self.args.audio_min_length:
                if self.args.drop_long and l > self.args.encodec_sr*self.args.audio_max_length:
                    continue
                self.data.append(d)
                self.lengths_list.append(l)
        logging.info(f"number of data points for {self.split} split: {len(self.lengths_list)}")

        # phoneme vocabulary
        vocab_fn = os.path.join(self.args.dataset_dir,"vocab.txt")
        shutil.copy(vocab_fn, os.path.join(self.args.exp_dir, "vocab.txt"))
        with open(vocab_fn, "r") as f:
            temp = [l.strip().split(" ") for l in f.readlines() if len(l) != 0]
            self.phn2num = {item[1]:int(item[0]) for item in temp}
        
        self.symbol_set = set(["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"])
    
    def __len__(self):
        return len(self.lengths_list)
    
    def _load_phn_enc(self, index):
        item = self.data[index]
        pf = os.path.join(self.args.dataset_dir, self.args.phn_folder_name, item[1]+".txt")
        ef = os.path.join(self.args.dataset_dir, self.args.encodec_folder_name, item[1]+".txt")
        try:
            with open(pf, "r") as p, open(ef, "r") as e:
                phns = [l.strip() for l in p.readlines()]
                assert len(phns) == 1, phns
                x = [self.phn2num[item] for item in phns[0].split(" ") if item not in self.symbol_set] # drop ["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"], as they are not in training set annotation
                encos = [l.strip().split() for k, l in enumerate(e.readlines()) if k < self.args.n_codebooks]
                
                assert len(encos) == self.args.n_codebooks, ef
                if self.args.special_first:
                    y = [[int(n)+self.args.n_special for n in l] for l in encos]
                else:
                    y = [[int(n) for n in l] for l in encos]
        except Exception as e:
            logging.info(f"loading failed for {pf} and {ef}, maybe files don't exist or are corrupted")
            logging.info(f"error message: {e}")
            return [], [[]]

        return x, y

    def __getitem__(self, index):
        x, y = self._load_phn_enc(index)
        x_len, y_len = len(x), len(y[0])

        if x_len == 0 or y_len == 0:
            return {
            "x": None, 
            "x_len": None, 
            "y": None, 
            "y_len": None, 
            "y_mask_interval": None, # index y_mask_interval[1] is the position of start_of_continue token
            "extra_mask_start": None # this is only used in VE1
            }
        while y_len < self.args.encodec_sr*self.args.audio_min_length:
            assert not self.args.dynamic_batching
            index = random.choice(range(len(self))) # regenerate an index
            x, y = self._load_phn_enc(index)
            x_len, y_len = len(x), len(y[0])
        if self.args.drop_long:
            while x_len > self.args.text_max_length or y_len > self.args.encodec_sr*self.args.audio_max_length:
                index = random.choice(range(len(self))) # regenerate an index
                x, y = self._load_phn_enc(index)
                x_len, y_len = len(x), len(y[0])

        ### padding and cropping below ###
        ### padding and cropping below ###
        # adjust the length of encodec codes, pad to max_len or randomly crop
        orig_y_len = copy.copy(y_len)
        max_len = int(self.args.audio_max_length * self.args.encodec_sr)
        if y_len > max_len:
            audio_start = random.choice(range(0, y_len-max_len))
            for i in range(len(y)):
                y[i] = y[i][audio_start:(audio_start+max_len)]
            y_len = max_len
        else:
            audio_start = 0
            if not self.args.dynamic_batching:
                pad = [0] * (max_len - y_len) if self.args.sep_special_token else [self.args.audio_pad_token] * (max_len - y_len)
                for i in range(len(y)):
                    y[i] = y[i] + pad
        
        # adjust text
        # if audio is cropped, and text is longer than max, crop max based on how audio is cropped
        if audio_start > 0 and len(x) > self.args.text_max_length: # if audio is longer than max and text is long than max, start text the way audio started
            x = x[int(len(x)*audio_start/orig_y_len):]
            if len(x) > self.args.text_max_length: # if text is still longer than max, cut the end
                x = x[:self.args.text_max_length]
        
        x_len = len(x)
        if x_len > self.args.text_max_length:
            text_start = random.choice(range(0, x_len - self.args.text_max_length))
            x = x[text_start:text_start+self.args.text_max_length]
            x_len = self.args.text_max_length
        elif self.args.pad_x and x_len <= self.args.text_max_length:
            pad = [0] * (self.args.text_max_length - x_len) if self.args.sep_special_token else [self.args.text_pad_token] * (self.args.text_max_length - x_len)
            x = x + pad
        ### padding and cropping above ###
        ### padding and cropping above ###

        return {
            "x": torch.LongTensor(x), 
            "x_len": x_len, 
            "y": torch.LongTensor(y), 
            "y_len": y_len
            }
            

    def collate(self, batch):
        out = {key:[] for key in batch[0]}
        for item in batch:
            if item['x'] == None: # deal with load failure
                continue
            for key, val in item.items():
                out[key].append(val)
        res = {}
        if self.args.pad_x:
            res["x"] = torch.stack(out["x"], dim=0)
        else:
            res["x"] = torch.nn.utils.rnn.pad_sequence(out["x"], batch_first=True, padding_value=self.args.text_pad_token)
        res["x_lens"] = torch.LongTensor(out["x_len"])
        if self.args.dynamic_batching:
            if out['y'][0].ndim==2:
                res['y'] = torch.nn.utils.rnn.pad_sequence([item.transpose(1,0) for item in out['y']],padding_value=self.args.audio_pad_token)
                res['y'] = res['y'].permute(1,2,0) # T B K -> B K T
            else:
                assert out['y'][0].ndim==1, out['y'][0].shape
                res['y'] = torch.nn.utils.rnn.pad_sequence(out['y'], batch_first=True, padding_value=self.args.audio_pad_token)
        else:
            res['y'] = torch.stack(out['y'], dim=0)
        res["y_lens"] = torch.LongTensor(out["y_len"])
        res["text_padding_mask"] = torch.arange(res['x'][0].shape[-1]).unsqueeze(0) >= res['x_lens'].unsqueeze(1)
        res["audio_padding_mask"] = torch.arange(res['y'][0].shape[-1]).unsqueeze(0) >= res['y_lens'].unsqueeze(1)
        return res