import argparse


def MyParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general training 
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--precision", type=str, default="float16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--tb_write_every_n_steps", type=int, default=100)
    parser.add_argument("--print_every_n_steps", type=int, default=400)
    parser.add_argument("--val_every_n_steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=100, help="this is the effective batch size, no matter whether using gradient_accumulation_steps, not used if we specified max_num_tokens")
    parser.add_argument("--max_num_tokens", type=int, default=100000, help="max number of encodec tokens per gpu, this is only used when using dynamic batching, will ignore batch size. Note this is the final effective batch size per GPU, i.e. gradient accumulated batch size per gpu")
    parser.add_argument("--val_max_num_tokens", type=int, default=None, help="FOR validation")
    parser.add_argument("--num_buckets", type=int, default=6, help='used for dynamic batching, bucketing the samples based on the number of tokens')
    parser.add_argument("--dynamic_batching", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_fraction", type=float, default=0.01, help="use linear warmup, the proportion of the training steps that are used for warming up")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=None, help="if not None, will ignore n_epochs and use num_steps as the total number of amount of training, can try e.g. 400000 i.e. 400k steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="the value for torch.nn.utils.clip_grad_norm_(), not used if we use ScaledAdam optimizer")
    parser.add_argument("--early_stop_step", type=int, default=3200, help="stop training after this many steps of non-improvement")
    parser.add_argument("--early_stop_threshold", type=float, default=-1.0, help="early stop after the improvement is below this threshold for certain number of steps")

    # optimizer focused
    parser.add_argument("--optimizer_name", type=str, default="AdamW", help="can also use ScaledAdam, in which case we'll also use the Eden scheduler")
    parser.add_argument("--reduce_lr_start_step", type=int, default=3000, help='after which significantly reduce the lr. a param for the eden optimizer')
    parser.add_argument("--pseudo_epoch_size", type=int, default=3000, help="only use for Eden scheduler.")
    parser.add_argument("--reduce_lr_start_epoch", type=int, default=4)
    parser.add_argument("--clipping_update_period", type=int, default=600)


    # path
    parser.add_argument("--exp_dir", type=str, default=None, help="will be combined with dataset name")
    parser.add_argument("--dataset", type=str, help="e.g. 'libritts', 'gigaspeech', they are folder name in the data dir also")
    parser.add_argument("--dataset_dir", type=str, help="need to be compatible with corresponding dataset py file")
    parser.add_argument("--phn_folder_name", type=str, default="phonemes", help="for libritts I also have arpa phns, in which case should be phonemes_arpa")
    parser.add_argument("--encodec_folder_name", type=str, default="encodec_16khz_4codebooks", help="folder where encodec codes are stored")
    parser.add_argument("--manifest_name", type=str, default="manifest", help="metadata filename")

    # data focused
    parser.add_argument("--pad_x", type=int, default=1, help="whether or not always pad x to have text_max_length. select 1 to get the maximal memory consumption, but the actual case should be smaller, better to have it being 0")
    parser.add_argument("--audio_max_length", type=float, default=20, help="in second, crop or drop the audio is length is longer than this")
    parser.add_argument("--audio_min_length", type=float, default=2, help="in second, drop the audio if length is shorter than this")
    parser.add_argument("--text_max_length", type=int, default=400, help='if too long, we crop or drop')
    parser.add_argument("--text_min_length", type=float, default=10, help="if too short, will drop")
    parser.add_argument("--encodec_sr", type=int, default=50, help="for my encodec that takes 16kHz audio with a downsample rate of 320, the codec sample rate is 50Hz, i.e. 50 codes (x n_codebooks) per second")
    parser.add_argument("--drop_long", type=int, default=0, help="if this is true, will drop example whose encodec sequence or phone sequence is too long, rather than cropping, to reduce hellucination")

    # encodec and token rearrangement
    parser.add_argument('--mask_len_min', type=int, default=1, help='Minimum mask length')
    parser.add_argument('--mask_len_max', type=int, default=600, help='Maximum mask length')
    parser.add_argument("--eos", type=int, default=-1, help="this is to be used with reduced_eog, where we end the utterance with eos, and end the generated segment with eog, also when this is used, the n_special should be 4")
    parser.add_argument("--reduced_eog", type=int, default=0, help="for the non-final segments, do not insert eog at the end, this could hopefully solve the early stopping issue when doing tts")
    parser.add_argument("--special_first", type=int, default=0, help="if 1, need to have special tokens to be the first few tokens, e.g. 0, 1, 2, which means we need to adjust the preprocessing and postprocessing of the encodec codes. note that we hard coded to have 3 special tokens")
    parser.add_argument("--n_special", type=int, default=3, help="empty, eog, pad, (eos)")
    parser.add_argument("--codebook_weight", type=str, default=None, help="e.g. ['5','1','0.5','0.1']")
    parser.add_argument("--max_mask_portion",type=float,default=0.7,help="should mask a utterance for more than this portion")
    parser.add_argument("--max_n_spans", type=int, default=3, help='maximal number of spans, only use when using multicm3, this is used to decide number of mask_embedding, and max clamp value if use Poisson distribution, if use uniform distribution to sample number of spans if will be uniform(1,max_n_spans)')
    parser.add_argument("--shuffle_mask_embedding", type=int, default=0, help="whether shuffle the mask embedding, so that mask:0 is not the most well trained, default is not shuffling. The default has it's benefit, as it make sure that mask:0 always appear the first")
    parser.add_argument("--mask_sample_dist", type=str, default="poisson1", help="uniform or poissonx, e.g. poisson1, meaning the parameter lambda is 1, it will most likely sample 1 masks")
    parser.add_argument("--min_gap", type=int, default=5, help="after sampled starts, delete later one if it closer to the former start than the min_gap")
    parser.add_argument('--n_codebooks', type=int, default=4)
    parser.add_argument('--text_vocab_size', type=int, default=100, help='Size of text vocabulary')
    parser.add_argument('--text_pad_token', type=int, default=100, help='padding of the text tokens, not attended')
    parser.add_argument('--audio_vocab_size', type=str, default='2048', help="Size of audio vocabulary")
    parser.add_argument("--empty_token", default=2048, type=int, help="indicating the no token at the position for the codebook")
    parser.add_argument('--eog', type=int, default=2049, help='End of generation token')
    parser.add_argument('--audio_pad_token', type=int, default=2050, help='padding of the encodec codes, not attended')

    # model focused
    parser.add_argument('--d_model', type=int, default=2048, help='Model dimension')
    parser.add_argument('--audio_embedding_dim', type=int, default=2048, help='dimension for encodec continues embedding (before being quantized)')
    parser.add_argument('--text_embedding_dropout', type=float, default=0.1, help='Dropout for text embedding')
    parser.add_argument('--audio_embedding_dropout', type=float, default=0, help='Dropout for audio embedding')
    parser.add_argument('--text_positional_embedding_dropout', type=float, default=0.1, help='Dropout for text positional embedding')
    parser.add_argument('--audio_positional_embedding_dropout', type=float, default=0.1, help='Dropout for audio positional embedding')
    parser.add_argument('--trm_dropout', type=float, default=0.1, help='Dropout for transformer')
    parser.add_argument('--nhead', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--num_decoder_layers', type=int, default=16, help='Number of decoder layers')
    parser.add_argument('--load_model_from', type=str, default=None, help='Path to load model from, this will be effective last, so will overwrite all previous load, including resume')
    return parser