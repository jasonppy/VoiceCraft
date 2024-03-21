import argparse, pickle
import logging
import os, random
import numpy as np
import torch
import torchaudio

from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text
)

from models import voicecraft
import argparse, time, tqdm

# this script only works for the musicgen architecture
def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest_fn", type=str, default="path/to/eval_metadata_file")
    parser.add_argument("--audio_root", type=str, default="path/to/audio_folder")
    parser.add_argument("--exp_dir", type=str, default="path/to/model_folder")
    parser.add_argument("--left_margin", type=float, default=0.08, help="extra space on the left to the word boundary")
    parser.add_argument("--right_margin", type=float, default=0.08, help="extra space on the right to the word boundary")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--codec_audio_sr", type=int, default=16000, help='the sample rate of audio that the codec is trained for')
    parser.add_argument("--codec_sr", type=int, default=50, help='the sample rate of the codec codes')
    parser.add_argument("--top_k", type=int, default=-1, help="sampling param")
    parser.add_argument("--top_p", type=float, default=0.8, help="sampling param")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling param")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--signature", type=str, default=None, help="path to the encodec model")
    parser.add_argument("--stop_repetition", type=int, default=2, help="used for inference, when the number of consecutive repetition of a token is bigger than this, stop it")
    parser.add_argument("--kvcache", type=int, default=1, help='if true, use kv cache, which is 4-8x faster than without')
    parser.add_argument("--silence_tokens", type=str, default="[1388,1898,131]", help="note that if you are not using the pretrained encodec 6f79c6a8, make sure you specified it yourself, rather than using the default")
    return parser.parse_args()

@torch.no_grad()
def inference_one_sample(model, model_args, phn2num, text_tokenizer, audio_tokenizer, audio_fn, target_text, mask_interval, device, decode_config):
    # phonemize
    text_tokens = [phn2num[phn] for phn in
            tokenize_text(
                text_tokenizer, text=target_text.strip()
            ) if phn in phn2num
        ]
    text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
    text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])

    encoded_frames = tokenize_audio(audio_tokenizer, audio_fn)
    original_audio = encoded_frames[0][0].transpose(2,1) # [1,T,K]
    assert original_audio.ndim==3 and original_audio.shape[0] == 1 and original_audio.shape[2] == model_args.n_codebooks, original_audio.shape
    logging.info(f"with direct encodec encoding before input, original audio length: {original_audio.shape[1]} codec frames, which is {original_audio.shape[1]/decode_config['codec_sr']:.2f} sec.")

    # forward
    stime = time.time()
    encoded_frames = model.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        original_audio[...,:model_args.n_codebooks].to(device), # [1,T,8]
        mask_interval=mask_interval.unsqueeze(0).to(device),
        top_k=decode_config['top_k'],
        top_p=decode_config['top_p'],
        temperature=decode_config['temperature'],
        stop_repetition=decode_config['stop_repetition'],
        kvcache=decode_config['kvcache'],
        silence_tokens=eval(decode_config['silence_tokens']) if type(decode_config['silence_tokens']) == str else decode_config['silence_tokens'],
    ) # output is [1,K,T]
    logging.info(f"inference on one sample take: {time.time() - stime:.4f} sec.")
    if type(encoded_frames) == tuple:
        encoded_frames = encoded_frames[0]
    logging.info(f"generated encoded_frames.shape: {encoded_frames.shape}, which is {encoded_frames.shape[-1]/decode_config['codec_sr']} sec.")
    

    # decode (both original and generated)
    original_sample = audio_tokenizer.decode(
        [(original_audio.transpose(2,1), None)] # [1,T,8] -> [1,8,T]
    )
    generated_sample = audio_tokenizer.decode(
        [(encoded_frames, None)]
    )

    return original_sample, generated_sample

def get_model(exp_dir, device=None):
    with open(os.path.join(exp_dir, "args.pkl"), "rb") as f:
        model_args = pickle.load(f)

    logging.info("load model weights...")
    model = voicecraft.VoiceCraft(model_args)
    ckpt_fn = os.path.join(exp_dir, "best_bundle.pth")
    ckpt = torch.load(ckpt_fn, map_location='cpu')['model']
    phn2num = torch.load(ckpt_fn, map_location='cpu')['phn2num']
    model.load_state_dict(ckpt)
    del ckpt
    logging.info("done loading weights...")
    if device == None:
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    return model, model_args, phn2num


def get_mask_interval(ali_fn, word_span_ind, editType):
    with open(ali_fn, "r") as rf:
        data = [l.strip().split(",") for l in rf.readlines()]
        data = data[1:]
    tmp = word_span_ind.split(",")
    s, e = int(tmp[0]), int(tmp[-1])
    start = None
    for j, item in enumerate(data):
        if j == s and item[3] == "words":
            if editType == 'insertion':
                start = float(item[1])
            else:
                start = float(item[0])
        if j == e and item[3] == "words":
            if editType == 'insertion':
                end = float(item[0])
            else:
                end = float(item[1])
            assert start != None
            break
    return (start, end)

if __name__ == "__main__":
    def seed_everything(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    # args.device = 'cpu'
    args.allowed_repeat_tokens = eval(args.allowed_repeat_tokens)
    seed_everything(args.seed)

    # load model
    stime = time.time()
    logging.info(f"loading model from {args.exp_dir}")
    model, model_args, phn2num = get_model(args.exp_dir)
    if not os.path.isfile(model_args.exp_dir):
        model_args.exp_dir = args.exp_dir
    logging.info(f"loading model done, took {time.time() - stime:.4f} sec")

    # setup text and audio tokenizer
    text_tokenizer = TextTokenizer(backend="espeak")
    audio_tokenizer = AudioTokenizer(signature=args.signature) # will also put the neural codec model on gpu

    with open(args.manifest_fn, "r") as rf:
        manifest = [l.strip().split("\t") for l in rf.readlines()]
    manifest = manifest[1:]
    
    # wav_fn	txt_fn	alingment_fn	num_words	word_span_ind
    audio_fns = []
    target_texts = []
    mask_intervals = []
    edit_types = []
    new_spans = []
    orig_spans = []
    os.makedirs(args.output_dir, exist_ok=True)
    if args.crop_concat:
        mfa_temp = f"{args.output_dir}/mfa_temp"
        os.makedirs(mfa_temp, exist_ok=True)
    for item in manifest:
        audio_fn = os.path.join(args.audio_root, item[0])
        temp = torchaudio.info(audio_fn)
        audio_dur = temp.num_frames/temp.sample_rate
        audio_fns.append(audio_fn)
        target_text = item[2].split("|")[-1]
        edit_types.append(item[5].split("|"))
        new_spans.append(item[4].split("|"))
        orig_spans.append(item[3].split("|"))
        target_texts.append(target_text) # the last transcript is the target
        # mi needs to be created from word_ind_span and alignment_fn, along with args.left_margin and args.right_margin
        mis = []
        all_ind_intervals = item[3].split("|")
        editTypes = item[5].split("|")
        smaller_indx = []
        alignment_fn = os.path.join(args.audio_root, "aligned", item[0].replace(".wav", ".csv"))
        if not os.path.isfile(alignment_fn):
            alignment_fn = alignment_fn.replace("/aligned/", "/aligned_csv/")
            assert os.path.isfile(alignment_fn), alignment_fn
        for ind_inter,editType in zip(all_ind_intervals, editTypes):
            # print(ind_inter)
            mi = get_mask_interval(alignment_fn, ind_inter, editType)
            mi = (max(mi[0] - args.left_margin, 1/args.codec_sr), min(mi[1] + args.right_margin, audio_dur)) # in seconds
            mis.append(mi)
            smaller_indx.append(mi[0])
        ind = np.argsort(smaller_indx)
        mis = [mis[id] for id in ind]
        mask_intervals.append(mis)



    for i, (audio_fn, target_text, mask_interval) in enumerate(tqdm.tqdm(zip(audio_fns, target_texts, mask_intervals))):
        orig_mask_interval = mask_interval
        mask_interval = [[round(cmi[0]*args.codec_sr), round(cmi[1]*args.codec_sr)] for cmi in mask_interval]
        # logging.info(f"i: {i}, mask_interval: {mask_interval}")
        mask_interval = torch.LongTensor(mask_interval) # [M,2]
        orig_audio, new_audio = inference_one_sample(model, model_args, phn2num, text_tokenizer, audio_tokenizer, audio_fn, target_text, mask_interval, args.device, vars(args))
        
        # save segments for comparison
        orig_audio, new_audio = orig_audio[0].cpu(), new_audio[0].cpu()
        # logging.info(f"length of the resynthesize orig audio: {orig_audio.shape}")

        save_fn_new = f"{args.output_dir}/{os.path.basename(audio_fn)[:-4]}_new_seed{args.seed}.wav"
        
        torchaudio.save(save_fn_new, new_audio, args.codec_audio_sr)

        save_fn_orig = f"{args.output_dir}/{os.path.basename(audio_fn)[:-4]}_orig.wav"
        if not os.path.isfile(save_fn_orig):
            orig_audio, orig_sr = torchaudio.load(audio_fn)
            if orig_sr != args.codec_audio_sr:
                orig_audio = torchaudio.transforms.Resample(orig_sr, args.codec_audio_sr)(orig_audio)
            torchaudio.save(save_fn_orig, orig_audio, args.codec_audio_sr)

