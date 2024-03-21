import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="encode the librilight dataset using encodec model")
    parser.add_argument("--manifest_root", type=str, default="/home/pyp/audiocraft/egs/gigaspeech", help="this the dir of the audiocraft manifest!")
    parser.add_argument('--audio_dir', type=str, default="/data/scratch/pyp/datasets/gigaspeech_flac", help="Path dirs of the flac audio files")
    parser.add_argument('--save_dir', type=str, default="/data/scratch/pyp/datasets/gigaspeech_phn_enc_manifest/xl", help="path to the manifest, phonemes, and encodec codes dirs")
    parser.add_argument('--encodec_model_path', type=str, default="/data/scratch/pyp/exp_pyp/audiocraft/encodec/xps/6f79c6a8/checkpoint.th")
    parser.add_argument('--n_workers', type=int, default=32, help="Number of parallel worker processes")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size for encodec encoding, decrease it if OOM. This is the sum of batch size *over each gpu*, so increase it if you are using more gpus")
    parser.add_argument('--model_sr', type=int, default=16000, help='encodec input audio sample rate')
    parser.add_argument('--downsample_rate', type=int, default=320, help='encodec downsample rate')
    parser.add_argument('--model_code_sr', type=int, default=50, help='encodec model code sample rate')
    parser.add_argument('--len_cap', type=float, default=35.0, help='will drop audios that are longer than this number')
    return parser.parse_args()

if __name__ == "__main__":
    import logging
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    import os
    import numpy as np
    import torch
    import torchaudio
    import tqdm
    import time

    args = parse_args()
    
    manifest_dir = args.manifest_root  # this dir is scp-ed
    audio_dir = args.audio_dir # this is scp-ed flac dir
    encodec_signature = args.encodec_model_path.split("/")[-2]
    save_codes_dir = os.path.join(args.save_dir, f"encodec_16khz_{encodec_signature}")
    os.makedirs(save_codes_dir, exist_ok=True)


    # model_sr = 16000
    # downsample_rate = 320
    # model_code_sr = 50
    def sort_by_audio_len(lens):
        inds = np.argsort(lens).tolist()
        logging.info(f"longest: {lens[inds[-1]]/args.downsample_rate} encodec codes, {lens[inds[-1]]/args.model_sr:.2f} sec.")
        logging.info(f"shortest: {lens[inds[0]]/args.downsample_rate} encodec codes, {lens[inds[0]]/args.model_sr:.2f} sec.")
        logging.info(f"median: {lens[inds[len(inds)//2]]/args.downsample_rate} encodec codes, {lens[inds[len(inds)//2]]/args.model_sr:.2f} sec.")
        logging.info(f"95 percentile longest: {lens[inds[int(len(inds)*0.95)]]/args.downsample_rate} encodec codes, {lens[inds[int(len(inds)*0.95)]]/args.model_sr:.2f} sec.")
        return inds[::-1]
    
    def write_array_to_txt_file(array, filename):
        with open(filename, 'w') as f:
            for a in array[:-1]:
                f.write(' '.join(map(str, a))+'\n')
            f.write(' '.join(map(str, array[-1])))

    

    class mydataset(torch.utils.data.Dataset):
        def __init__(self, split):
            super().__init__()
            # self.data = gs[split]
            self.split = split
            self.audio_root = audio_dir
            manifest_fn = os.path.join(manifest_dir, split+".txt")
            with open(manifest_fn, "r") as rf:
                self.data = [l.strip().split("\t") for l in rf.readlines()]
        def __len__(self):
            return len(self.data)
        def __getitem__(self, ind):
            try:
                afn = self.data[ind][0]
                fn = os.path.join(self.audio_root, afn)
                audio, sr = torchaudio.load(fn)
                assert sr == args.model_sr, sr
            except Exception as e:
                logging.info(f"{e}")
                return None, None, None
            assert audio.ndim==2 and audio.shape[0] == 1, audio.shape
            return audio.type(torch.float32).squeeze(0), audio.shape[-1], os.path.basename(afn).split(".")[0]
        def collate(self, batch):
            lens, audios, segment_ids = [], [], []
            for item in batch:
                if item[0] != None:
                    audios.append(item[0])
                    lens.append(item[1])
                    segment_ids.append(item[2])
            return audios, lens, segment_ids

    # load the encodec model
    from audiocraft.solvers import CompressionSolver
    model = CompressionSolver.model_from_checkpoint(args.encodec_model_path)
    model = model.cuda()
    model = model.eval()
    model = torch.nn.DataParallel(model)


    # setup dataloader
    mega_batch_size = 2100
    batch_size = args.batch_size
    train_dataset = mydataset('train')
    train_loader = torch.torch.utils.data.DataLoader(train_dataset, batch_size=mega_batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers, collate_fn=train_dataset.collate)
    validation_dataset = mydataset('validation')
    validation_loader = torch.torch.utils.data.DataLoader(validation_dataset, batch_size=mega_batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers, collate_fn=validation_dataset.collate)
    test_dataset = mydataset('test')
    test_loader = torch.torch.utils.data.DataLoader(test_dataset, batch_size=mega_batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers, collate_fn=test_dataset.collate)
    splits = ['validation', 'test', 'train']
    loaders = [validation_loader, test_loader, train_loader]
    # splits = ['validation'] # NOTE this is for debug, for example, see if the 
    # loaders = [validation_loader]
    for split, loader in zip(splits, loaders):
        skip = 0
        logging.info(f"now processing split {split}...")
        mega_n_steps = int(np.ceil(len(loader.dataset) / mega_batch_size))
        # mega_n_steps = int(np.ceil(len(gs) / mega_batch_size))
        logging.info(f"partition the split {split} into {mega_n_steps} parts, each has {mega_batch_size} samples")
        # with open(mani_fn, "a") as mani_wf: # resume from where we failed
        for m, mega_batch in enumerate(loader):
            logging.info(f"====================================")
            logging.info(f"====================================")
            logging.info(f"now processing mega step {m+1}/{mega_n_steps}")
            lengths = np.array(mega_batch[1])
            sorted_inds = sort_by_audio_len(lengths)
            for j in range(len(sorted_inds))[::-1]:
                if lengths[sorted_inds[j]] < args.model_sr*0.2 or lengths[sorted_inds[j]] > args.model_sr*args.len_cap: # skip samples that are too short (shorter than 0.2s), or too big (bigger than 80s)
                    skip += 1
                    del sorted_inds[j]
            
            n_steps = int(np.ceil(len(sorted_inds) / batch_size))
            for n in tqdm.tqdm(range(n_steps), disable=True):
                inds_used = sorted_inds[n*batch_size:(n+1)*batch_size]
                wav_batch = [mega_batch[0][id] for id in inds_used]
                all_lens = [mega_batch[1][id] for id in inds_used]
                segment_id_batch = [mega_batch[2][id] for id in inds_used]
                # print(segment_id_batch)
                padded_wav = torch.nn.utils.rnn.pad_sequence(wav_batch, batch_first=True).unsqueeze(1) # [B, T] -> [B, 1, T]
                with torch.no_grad():
                    if max(all_lens) > 300000 and len(all_lens) > 1: # NOTE decrease this (300000) if OOM, or chunk it into more than 2 forward passes
                        codes = []
                        inwav = padded_wav.cuda()
                        codes.append(model(inwav[:len(inwav)//2], encode=True)[0].cpu())
                        codes.append(model(inwav[len(inwav)//2:], encode=True)[0].cpu())
                        codes = torch.cat(codes, dim=0)
                    else:
                        encoded_frames = model(padded_wav.cuda(), encode=True) # wav needs to have shape [B, C, T], C is model.channels, which is 1 for the 24kHz encodec model
                        # logging.info(f"encoded_frames: {encoded_frames[0].shape}")
                        codes = encoded_frames[0].cpu()

                for i, length in enumerate(all_lens):
                    save_fn = os.path.join(save_codes_dir, segment_id_batch[i]+".txt")
                    actual_len = round(length / args.downsample_rate) # 320 is downsample rate for this model
                    cur_code = codes[i].tolist() if type(codes) == list else codes[i, :, :actual_len].tolist()
                    write_array_to_txt_file(cur_code, save_fn)

                    # mani_wf.write(f"0\t{segment_id_batch[i]}\t{len(cur_code[0])}\n") # write to manifest file
                    # if i == 10:
                    #    raise
            # break
        # logging.info(f"split {split} has {len(gs[split])} samples in total, skipped {skip} due to forbiden words")
        logging.info(f"split {split} has {len(loader.dataset)} samples in total, skipped {skip} due to utterance being too long or too short")
        # break