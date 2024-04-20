import os
import gradio as gr
import torch
import torchaudio
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
from models import voicecraft
import io
import numpy as np
import random
import uuid


DEMO_PATH = os.getenv("DEMO_PATH", "./demo")
TMP_PATH = os.getenv("TMP_PATH", "./demo/temp")
MODELS_PATH = os.getenv("MODELS_PATH", "./pretrained_models")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model, align_model, voicecraft_model = None, None, None


def get_random_string():
    return "".join(str(uuid.uuid4()).split("-"))


def seed_everything(seed):
    if seed != -1:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class WhisperxAlignModel:
    def __init__(self):
        from whisperx import load_align_model
        self.model, self.metadata = load_align_model(language_code="en", device=device)

    def align(self, segments, audio_path):
        from whisperx import align, load_audio
        audio = load_audio(audio_path)
        return align(segments, self.model, self.metadata, audio, device, return_char_alignments=False)["segments"]


class WhisperModel:
    def __init__(self, model_name):
        from whisper import load_model
        self.model = load_model(model_name, device)

        from whisper.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(multilingual=False)
        self.supress_tokens = [-1] + [
            i
            for i in range(tokenizer.eot)
            if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" "))
        ]

    def transcribe(self, audio_path):
        return self.model.transcribe(audio_path, suppress_tokens=self.supress_tokens, word_timestamps=True)["segments"]


class WhisperxModel:
    def __init__(self, model_name, align_model: WhisperxAlignModel):
        from whisperx import load_model
        self.model = load_model(model_name, device, asr_options={"suppress_numerals": True, "max_new_tokens": None, "clip_timestamps": None, "hallucination_silence_threshold": None})
        self.align_model = align_model

    def transcribe(self, audio_path):
        segments = self.model.transcribe(audio_path, batch_size=8)["segments"]
        return self.align_model.align(segments, audio_path)


def load_models(whisper_backend_name, whisper_model_name, alignment_model_name, voicecraft_model_name):
    global transcribe_model, align_model, voicecraft_model

    if voicecraft_model_name == "giga330M_TTSEnhanced":
        voicecraft_model_name = "gigaHalfLibri330M_TTSEnhanced_max16s"

    if alignment_model_name is not None:
        align_model = WhisperxAlignModel()

    if whisper_model_name is not None:
        if whisper_backend_name == "whisper":
            transcribe_model = WhisperModel(whisper_model_name)
        else:
            if align_model is None:
                raise gr.Error("Align model required for whisperx backend")
            transcribe_model = WhisperxModel(whisper_model_name, align_model)

    voicecraft_name = f"{voicecraft_model_name}.pth"
    model = voicecraft.VoiceCraft.from_pretrained(f"pyp1/VoiceCraft_{voicecraft_name.replace('.pth', '')}")
    phn2num = model.args.phn2num
    config = model.args
    model.to(device)

    encodec_fn = f"{MODELS_PATH}/encodec_4cb2048_giga.th"
    if not os.path.exists(encodec_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th -O " + encodec_fn)

    voicecraft_model = {
        "config": config,
        "phn2num": phn2num,
        "model": model,
        "text_tokenizer": TextTokenizer(backend="espeak"),
        "audio_tokenizer": AudioTokenizer(signature=encodec_fn)
    }
    return gr.Accordion()


def get_transcribe_state(segments):
    words_info = [word_info for segment in segments for word_info in segment["words"]]
    transcript = " ".join([segment["text"] for segment in segments])
    transcript = transcript[1:] if transcript[0] == " " else transcript
    return {
        "segments": segments,
        "transcript": transcript,
        "words_info": words_info,
        "transcript_with_start_time": " ".join([f"{word['start']} {word['word']}" for word in words_info]),
        "transcript_with_end_time": " ".join([f"{word['word']} {word['end']}" for word in words_info]),
        "word_bounds": [f"{word['start']} {word['word']} {word['end']}" for word in words_info]
    }


def transcribe(seed, audio_path):
    if transcribe_model is None:
        raise gr.Error("Transcription model not loaded")
    seed_everything(seed)

    segments = transcribe_model.transcribe(audio_path)
    state = get_transcribe_state(segments)

    return [
        state["transcript"], state["transcript_with_start_time"], state["transcript_with_end_time"],
        gr.Dropdown(value=state["word_bounds"][-1], choices=state["word_bounds"], interactive=True), # prompt_to_word
        gr.Dropdown(value=state["word_bounds"][0], choices=state["word_bounds"], interactive=True), # edit_from_word
        gr.Dropdown(value=state["word_bounds"][-1], choices=state["word_bounds"], interactive=True), # edit_to_word
        state
    ]


def align_segments(transcript, audio_path):
    from aeneas.executetask import ExecuteTask
    from aeneas.task import Task
    import json
    config_string = 'task_language=eng|os_task_file_format=json|is_text_type=plain'

    tmp_transcript_path = os.path.join(TMP_PATH, f"{get_random_string()}.txt")
    tmp_sync_map_path = os.path.join(TMP_PATH, f"{get_random_string()}.json")
    with open(tmp_transcript_path, "w") as f:
        f.write(transcript)

    task = Task(config_string=config_string)
    task.audio_file_path_absolute = os.path.abspath(audio_path)
    task.text_file_path_absolute = os.path.abspath(tmp_transcript_path)
    task.sync_map_file_path_absolute = os.path.abspath(tmp_sync_map_path)
    ExecuteTask(task).execute()
    task.output_sync_map_file()

    with open(tmp_sync_map_path, "r") as f:
        return json.load(f)


def align(seed, transcript, audio_path):
    if align_model is None:
        raise gr.Error("Align model not loaded")
    seed_everything(seed)

    fragments = align_segments(transcript, audio_path)
    segments = [{
        "start": float(fragment["begin"]),
        "end": float(fragment["end"]),
        "text": " ".join(fragment["lines"])
    } for fragment in fragments["fragments"]]
    segments = align_model.align(segments, audio_path)
    state = get_transcribe_state(segments)

    return [
        state["transcript_with_start_time"], state["transcript_with_end_time"],
        gr.Dropdown(value=state["word_bounds"][-1], choices=state["word_bounds"], interactive=True), # prompt_to_word
        gr.Dropdown(value=state["word_bounds"][0], choices=state["word_bounds"], interactive=True), # edit_from_word
        gr.Dropdown(value=state["word_bounds"][-1], choices=state["word_bounds"], interactive=True), # edit_to_word
        state
    ]


def get_output_audio(audio_tensors, codec_audio_sr):
    result = torch.cat(audio_tensors, 1)
    buffer = io.BytesIO()
    torchaudio.save(buffer, result, int(codec_audio_sr), format="wav")
    buffer.seek(0)
    return buffer.read()


def run(seed, left_margin, right_margin, codec_audio_sr, codec_sr, top_k, top_p, temperature,
        stop_repetition, sample_batch_size, kvcache, silence_tokens,
        audio_path, transcribe_state, transcript, smart_transcript,
        mode, prompt_end_time, edit_start_time, edit_end_time,
        split_text, selected_sentence, previous_audio_tensors):
    if voicecraft_model is None:
        raise gr.Error("VoiceCraft model not loaded")
    if smart_transcript and (transcribe_state is None):
        raise gr.Error("Can't use smart transcript: whisper transcript not found")

    seed_everything(seed)
    if mode == "Long TTS":
        if split_text == "Newline":
            sentences = transcript.split('\n')
        else:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(transcript.replace("\n", " "))
    elif mode == "Rerun":
        colon_position = selected_sentence.find(':')
        selected_sentence_idx = int(selected_sentence[:colon_position])
        sentences = [selected_sentence[colon_position + 1:]]
    else:
        sentences = [transcript.replace("\n", " ")]

    info = torchaudio.info(audio_path)
    audio_dur = info.num_frames / info.sample_rate

    audio_tensors = []
    inference_transcript = ""
    for sentence in sentences:
        decode_config = {"top_k": top_k, "top_p": top_p, "temperature": temperature, "stop_repetition": stop_repetition,
                         "kvcache": kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr,
                         "silence_tokens": silence_tokens, "sample_batch_size": sample_batch_size}
        if mode != "Edit":
            from inference_tts_scale import inference_one_sample

            if smart_transcript:
                target_transcript = ""
                for word in transcribe_state["words_info"]:
                    if word["end"] < prompt_end_time:
                        target_transcript += word["word"] + (" " if word["word"][-1] != " " else "")
                    elif (word["start"] + word["end"]) / 2 < prompt_end_time:
                        # include part of the word it it's big, but adjust prompt_end_time
                        target_transcript += word["word"] + (" " if word["word"][-1] != " " else "")
                        prompt_end_time = word["end"]
                        break
                    else:
                        break
                target_transcript += f" {sentence}"
            else:
                target_transcript = sentence

            inference_transcript += target_transcript + "\n"

            prompt_end_frame = int(min(audio_dur, prompt_end_time) * info.sample_rate)
            _, gen_audio = inference_one_sample(voicecraft_model["model"],
                                                voicecraft_model["config"],
                                                voicecraft_model["phn2num"],
                                                voicecraft_model["text_tokenizer"], voicecraft_model["audio_tokenizer"],
                                                audio_path, target_transcript, device, decode_config,
                                                prompt_end_frame)
        else:
            from inference_speech_editing_scale import inference_one_sample

            if smart_transcript:
                target_transcript = ""
                for word in transcribe_state["words_info"]:
                    if word["start"] < edit_start_time:
                        target_transcript += word["word"] + (" " if word["word"][-1] != " " else "")
                    else:
                        break
                target_transcript += f" {sentence}"
                for word in transcribe_state["words_info"]:
                    if word["end"] > edit_end_time:
                        target_transcript += word["word"] + (" " if word["word"][-1] != " " else "")
            else:
                target_transcript = sentence

            inference_transcript += target_transcript + "\n"

            morphed_span = (max(edit_start_time - left_margin, 1 / codec_sr), min(edit_end_time + right_margin, audio_dur))
            mask_interval = [[round(morphed_span[0]*codec_sr), round(morphed_span[1]*codec_sr)]]
            mask_interval = torch.LongTensor(mask_interval)

            _, gen_audio = inference_one_sample(voicecraft_model["model"],
                                                voicecraft_model["config"],
                                                voicecraft_model["phn2num"],
                                                voicecraft_model["text_tokenizer"], voicecraft_model["audio_tokenizer"],
                                                audio_path, target_transcript, mask_interval, device, decode_config)
        gen_audio = gen_audio[0].cpu()
        audio_tensors.append(gen_audio)

    if mode != "Rerun":
        output_audio = get_output_audio(audio_tensors, codec_audio_sr)
        sentences = [f"{idx}: {text}" for idx, text in enumerate(sentences)]
        component = gr.Dropdown(choices=sentences, value=sentences[0])
        return output_audio, inference_transcript, component, audio_tensors
    else:
        previous_audio_tensors[selected_sentence_idx] = audio_tensors[0]
        output_audio = get_output_audio(previous_audio_tensors, codec_audio_sr)
        sentence_audio = get_output_audio(audio_tensors, codec_audio_sr)
        return output_audio, inference_transcript, sentence_audio, previous_audio_tensors


def update_input_audio(audio_path):
    if audio_path is None:
        return 0, 0, 0

    info = torchaudio.info(audio_path)
    max_time = round(info.num_frames / info.sample_rate, 2)
    return [
        gr.Slider(maximum=max_time, value=max_time),
        gr.Slider(maximum=max_time, value=0),
        gr.Slider(maximum=max_time, value=max_time),
    ]


def change_mode(mode):
    # tts_mode_controls, edit_mode_controls, edit_word_mode, split_text, long_tts_sentence_editor
    return [
        gr.Group(visible=mode != "Edit"),
        gr.Group(visible=mode == "Edit"),
        gr.Radio(visible=mode == "Edit"),
        gr.Radio(visible=mode == "Long TTS"),
        gr.Group(visible=mode == "Long TTS"),
    ]


def load_sentence(selected_sentence, codec_audio_sr, audio_tensors):
    if selected_sentence is None:
        return None
    colon_position = selected_sentence.find(':')
    selected_sentence_idx = int(selected_sentence[:colon_position])
    return get_output_audio([audio_tensors[selected_sentence_idx]], codec_audio_sr)


def update_bound_word(is_first_word, selected_word, edit_word_mode):
    if selected_word is None:
        return None

    word_start_time = float(selected_word.split(' ')[0])
    word_end_time = float(selected_word.split(' ')[-1])
    if edit_word_mode == "Replace half":
        bound_time = (word_start_time + word_end_time) / 2
    elif is_first_word:
        bound_time = word_start_time
    else:
        bound_time = word_end_time

    return bound_time


def update_bound_words(from_selected_word, to_selected_word, edit_word_mode):
    return [
        update_bound_word(True, from_selected_word, edit_word_mode),
        update_bound_word(False, to_selected_word, edit_word_mode),
    ]


smart_transcript_info = """
If enabled, the target transcript will be constructed for you:</br>
 - In TTS and Long TTS mode just write the text you want to synthesize.</br>
 - In Edit mode just write the text to replace selected editing segment.</br>
If disabled, you should write the target transcript yourself:</br>
 - In TTS mode write prompt transcript followed by generation transcript.</br>
 - In Long TTS select split by newline (<b>SENTENCE SPLIT WON'T WORK</b>) and start each line with a prompt transcript.</br>
 - In Edit mode write full prompt</br>
"""

demo_original_transcript = "But when I had approached so near to them, the common object, which the sense deceives, lost not by distance any of its marks."

demo_text = {
    "TTS": {
        "smart": "I cannot believe that the same model can also do text to speech synthesis too!",
        "regular": "But when I had approached so near to them, the common I cannot believe that the same model can also do text to speech synthesis too!"
    },
    "Edit": {
        "smart": "saw the mirage of the lake in the distance,",
        "regular": "But when I saw the mirage of the lake in the distance, which the sense deceives, Lost not by distance any of its marks,"
    },
    "Long TTS": {
        "smart": "You can run the model on a big text!\n"
                 "Just write it line-by-line. Or sentence-by-sentence.\n"
                 "If some sentences sound odd, just rerun the model on them, no need to generate the whole text again!",
        "regular": "But when I had approached so near to them, the common You can run the model on a big text!\n"
                   "But when I had approached so near to them, the common Just write it line-by-line. Or sentence-by-sentence.\n"
                   "But when I had approached so near to them, the common If some sentences sound odd, just rerun the model on them, no need to generate the whole text again!"
    }
}

all_demo_texts = {vv for k, v in demo_text.items() for kk, vv in v.items()}

demo_words = [
    '0.029 But 0.149', '0.189 when 0.33', '0.43 I 0.49', '0.53 had 0.65', '0.711 approached 1.152', '1.352 so 1.593',
    '1.693 near 1.933', '1.994 to 2.074', '2.134 them, 2.354', '2.535 the 2.655', '2.695 common 3.016', '3.196 object, 3.577',
    '3.717 which 3.898', '3.958 the 4.058', '4.098 sense 4.359', '4.419 deceives, 4.92', '5.101 lost 5.481', '5.682 not 5.963',
    '6.043 by 6.183', '6.223 distance 6.644', '6.905 any 7.065', '7.125 of 7.185', '7.245 its 7.346', '7.406 marks. 7.727'
]

demo_words_info = [
    {'word': 'But', 'start': 0.029, 'end': 0.149, 'score': 0.834}, {'word': 'when', 'start': 0.189, 'end': 0.33, 'score': 0.879},
    {'word': 'I', 'start': 0.43, 'end': 0.49, 'score': 0.984}, {'word': 'had', 'start': 0.53, 'end': 0.65, 'score': 0.998},
    {'word': 'approached', 'start': 0.711, 'end': 1.152, 'score': 0.822}, {'word': 'so', 'start': 1.352, 'end': 1.593, 'score': 0.822},
    {'word': 'near', 'start': 1.693, 'end': 1.933, 'score': 0.752}, {'word': 'to', 'start': 1.994, 'end': 2.074, 'score': 0.924},
    {'word': 'them,', 'start': 2.134, 'end': 2.354, 'score': 0.914}, {'word': 'the', 'start': 2.535, 'end': 2.655, 'score': 0.818},
    {'word': 'common', 'start': 2.695, 'end': 3.016, 'score': 0.971}, {'word': 'object,', 'start': 3.196, 'end': 3.577, 'score': 0.823},
    {'word': 'which', 'start': 3.717, 'end': 3.898, 'score': 0.701}, {'word': 'the', 'start': 3.958, 'end': 4.058, 'score': 0.798},
    {'word': 'sense', 'start': 4.098, 'end': 4.359, 'score': 0.797}, {'word': 'deceives,', 'start': 4.419, 'end': 4.92, 'score': 0.802},
    {'word': 'lost', 'start': 5.101, 'end': 5.481, 'score': 0.71}, {'word': 'not', 'start': 5.682, 'end': 5.963, 'score': 0.781},
    {'word': 'by', 'start': 6.043, 'end': 6.183, 'score': 0.834}, {'word': 'distance', 'start': 6.223, 'end': 6.644, 'score': 0.899},
    {'word': 'any', 'start': 6.905, 'end': 7.065, 'score': 0.893}, {'word': 'of', 'start': 7.125, 'end': 7.185, 'score': 0.772},
    {'word': 'its', 'start': 7.245, 'end': 7.346, 'score': 0.778}, {'word': 'marks.', 'start': 7.406, 'end': 7.727, 'score': 0.955}
]


def update_demo(mode, smart_transcript, edit_word_mode, transcript, edit_from_word, edit_to_word):
    if transcript not in all_demo_texts:
        return transcript, edit_from_word, edit_to_word

    replace_half = edit_word_mode == "Replace half"
    change_edit_from_word = edit_from_word == demo_words[2] or edit_from_word == demo_words[3]
    change_edit_to_word = edit_to_word == demo_words[11] or edit_to_word == demo_words[12]
    demo_edit_from_word_value = demo_words[2] if replace_half else demo_words[3]
    demo_edit_to_word_value = demo_words[12] if replace_half else demo_words[11]
    return [
        demo_text[mode]["smart" if smart_transcript else "regular"],
        demo_edit_from_word_value if change_edit_from_word else edit_from_word,
        demo_edit_to_word_value if change_edit_to_word else edit_to_word,
    ]


def get_app():
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column(scale=2):
                load_models_btn = gr.Button(value="Load models")
            with gr.Column(scale=5):
                with gr.Accordion("Select models", open=False) as models_selector:
                    with gr.Row():
                        voicecraft_model_choice = gr.Radio(label="VoiceCraft model", value="giga830M",
                                                        choices=["giga330M", "giga830M", "giga330M_TTSEnhanced"])
                        whisper_backend_choice = gr.Radio(label="Whisper backend", value="whisperX", choices=["whisper", "whisperX"])
                        whisper_model_choice = gr.Radio(label="Whisper model", value="base.en",
                                                        choices=[None, "base.en", "small.en", "medium.en", "large"])
                        align_model_choice = gr.Radio(label="Forced alignment model", value="whisperX", choices=[None, "whisperX"])

        with gr.Row():
            with gr.Column(scale=2):
                input_audio = gr.Audio(value=f"{DEMO_PATH}/84_121550_000074_000000.wav", label="Input Audio", type="filepath", interactive=True)
                with gr.Group():
                    original_transcript = gr.Textbox(label="Original transcript", lines=5, value=demo_original_transcript,
                                                    info="Use whisper model to get the transcript. Fix and align it if necessary.")
                    with gr.Accordion("Word start time", open=False):
                        transcript_with_start_time = gr.Textbox(label="Start time", lines=5, interactive=False, info="Start time before each word")
                    with gr.Accordion("Word end time", open=False):
                        transcript_with_end_time = gr.Textbox(label="End time", lines=5, interactive=False, info="End time after each word")

                    transcribe_btn = gr.Button(value="Transcribe")
                    align_btn = gr.Button(value="Align")

            with gr.Column(scale=3):
                with gr.Group():
                    transcript = gr.Textbox(label="Text", lines=7, value=demo_text["TTS"]["smart"])
                    with gr.Row():
                        smart_transcript = gr.Checkbox(label="Smart transcript", value=True)
                        with gr.Accordion(label="?", open=False):
                            info = gr.Markdown(value=smart_transcript_info)

                    with gr.Row():
                        mode = gr.Radio(label="Mode", choices=["TTS", "Edit", "Long TTS"], value="TTS")
                        split_text = gr.Radio(label="Split text", choices=["Newline", "Sentence"], value="Newline",
                                            info="Split text into parts and run TTS for each part.", visible=False)
                        edit_word_mode = gr.Radio(label="Edit word mode", choices=["Replace half", "Replace all"], value="Replace half",
                                                info="What to do with first and last word", visible=False)

                    with gr.Group() as tts_mode_controls:
                        prompt_to_word = gr.Dropdown(label="Last word in prompt", choices=demo_words, value=demo_words[10], interactive=True)
                        prompt_end_time = gr.Slider(label="Prompt end time", minimum=0, maximum=7.93, step=0.001, value=3.016)

                    with gr.Group(visible=False) as edit_mode_controls:
                        with gr.Row():
                            edit_from_word = gr.Dropdown(label="First word to edit", choices=demo_words, value=demo_words[2], interactive=True)
                            edit_to_word = gr.Dropdown(label="Last word to edit", choices=demo_words, value=demo_words[12], interactive=True)
                        with gr.Row():
                            edit_start_time = gr.Slider(label="Edit from time", minimum=0, maximum=7.93, step=0.001, value=0.46)
                            edit_end_time = gr.Slider(label="Edit to time", minimum=0, maximum=7.93, step=0.001, value=3.808)

                    run_btn = gr.Button(value="Run")

            with gr.Column(scale=2):
                output_audio = gr.Audio(label="Output Audio")
                with gr.Accordion("Inference transcript", open=False):
                    inference_transcript = gr.Textbox(label="Inference transcript", lines=5, interactive=False,
                                                    info="Inference was performed on this transcript.")
                with gr.Group(visible=False) as long_tts_sentence_editor:
                    sentence_selector = gr.Dropdown(label="Sentence", value=None,
                                                    info="Select sentence you want to regenerate")
                    sentence_audio = gr.Audio(label="Sentence Audio", scale=2)
                    rerun_btn = gr.Button(value="Rerun")

        with gr.Row():
            with gr.Accordion("Generation Parameters - change these if you are unhappy with the generation", open=False):
                stop_repetition = gr.Radio(label="stop_repetition", choices=[-1, 1, 2, 3, 4], value=3,
                                        info="if there are long silence in the generated audio, reduce the stop_repetition to 2 or 1. -1 = disabled")
                sample_batch_size = gr.Number(label="speech rate", value=4, precision=0,
                                            info="The higher the number, the faster the output will be. "
                                                "Under the hood, the model will generate this many samples and choose the shortest one. "
                                                "For giga330M_TTSEnhanced, 1 or 2 should be fine since the model is trained to do TTS.")
                seed = gr.Number(label="seed", value=-1, precision=0, info="random seeds always works :)")
                kvcache = gr.Radio(label="kvcache", choices=[0, 1], value=1,
                                    info="set to 0 to use less VRAM, but with slower inference")
                left_margin = gr.Number(label="left_margin", value=0.08, info="margin to the left of the editing segment")
                right_margin = gr.Number(label="right_margin", value=0.08, info="margin to the right of the editing segment")
                top_p = gr.Number(label="top_p", value=0.9, info="0.9 is a good value, 0.8 is also good")
                temperature = gr.Number(label="temperature", value=1, info="haven't try other values, do not recommend to change")
                top_k = gr.Number(label="top_k", value=0, info="0 means we don't use topk sampling, because we use topp sampling")
                codec_audio_sr = gr.Number(label="codec_audio_sr", value=16000, info='encodec specific, Do not change')
                codec_sr = gr.Number(label="codec_sr", value=50, info='encodec specific, Do not change')
                silence_tokens = gr.Textbox(label="silence tokens", value="[1388,1898,131]", info="encodec specific, do not change")


        audio_tensors = gr.State()
        transcribe_state = gr.State(value={"words_info": demo_words_info})


        mode.change(fn=update_demo,
                    inputs=[mode, smart_transcript, edit_word_mode, transcript, edit_from_word, edit_to_word],
                    outputs=[transcript, edit_from_word, edit_to_word])
        edit_word_mode.change(fn=update_demo,
                            inputs=[mode, smart_transcript, edit_word_mode, transcript, edit_from_word, edit_to_word],
                            outputs=[transcript, edit_from_word, edit_to_word])
        smart_transcript.change(fn=update_demo,
                                inputs=[mode, smart_transcript, edit_word_mode, transcript, edit_from_word, edit_to_word],
                                outputs=[transcript, edit_from_word, edit_to_word])

        load_models_btn.click(fn=load_models,
                            inputs=[whisper_backend_choice, whisper_model_choice, align_model_choice, voicecraft_model_choice],
                            outputs=[models_selector])

        input_audio.upload(fn=update_input_audio,
                        inputs=[input_audio],
                        outputs=[prompt_end_time, edit_start_time, edit_end_time])
        transcribe_btn.click(fn=transcribe,
                            inputs=[seed, input_audio],
                            outputs=[original_transcript, transcript_with_start_time, transcript_with_end_time,
                                    prompt_to_word, edit_from_word, edit_to_word, transcribe_state])
        align_btn.click(fn=align,
                        inputs=[seed, original_transcript, input_audio],
                        outputs=[transcript_with_start_time, transcript_with_end_time,
                                prompt_to_word, edit_from_word, edit_to_word, transcribe_state])

        mode.change(fn=change_mode,
                    inputs=[mode],
                    outputs=[tts_mode_controls, edit_mode_controls, edit_word_mode, split_text, long_tts_sentence_editor])

        run_btn.click(fn=run,
                    inputs=[
                        seed, left_margin, right_margin,
                        codec_audio_sr, codec_sr,
                        top_k, top_p, temperature,
                        stop_repetition, sample_batch_size,
                        kvcache, silence_tokens,
                        input_audio, transcribe_state, transcript, smart_transcript,
                        mode, prompt_end_time, edit_start_time, edit_end_time,
                        split_text, sentence_selector, audio_tensors
                    ],
                    outputs=[output_audio, inference_transcript, sentence_selector, audio_tensors])

        sentence_selector.change(fn=load_sentence,
                                inputs=[sentence_selector, codec_audio_sr, audio_tensors],
                                outputs=[sentence_audio])
        rerun_btn.click(fn=run,
                        inputs=[
                            seed, left_margin, right_margin,
                            codec_audio_sr, codec_sr,
                            top_k, top_p, temperature,
                            stop_repetition, sample_batch_size,
                            kvcache, silence_tokens,
                            input_audio, transcribe_state, transcript, smart_transcript,
                            gr.State(value="Rerun"), prompt_end_time, edit_start_time, edit_end_time,
                            split_text, sentence_selector, audio_tensors
                        ],
                        outputs=[output_audio, inference_transcript, sentence_audio, audio_tensors])

        prompt_to_word.change(fn=update_bound_word,
                            inputs=[gr.State(False), prompt_to_word, gr.State("Replace all")],
                            outputs=[prompt_end_time])
        edit_from_word.change(fn=update_bound_word,
                            inputs=[gr.State(True), edit_from_word, edit_word_mode],
                            outputs=[edit_start_time])
        edit_to_word.change(fn=update_bound_word,
                            inputs=[gr.State(False), edit_to_word, edit_word_mode],
                            outputs=[edit_end_time])
        edit_word_mode.change(fn=update_bound_words,
                            inputs=[edit_from_word, edit_to_word, edit_word_mode],
                            outputs=[edit_start_time, edit_end_time])
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VoiceCraft gradio app.")
    
    parser.add_argument("--demo-path", default="./demo", help="Path to demo directory")
    parser.add_argument("--tmp-path", default="./demo/temp", help="Path to tmp directory")
    parser.add_argument("--models-path", default="./pretrained_models", help="Path to voicecraft models directory")
    parser.add_argument("--port", default=7860, type=int, help="App port")
    parser.add_argument("--share", action="store_true", help="Launch with public url")
    parser.add_argument("--server_name", default="127.0.0.1", type=str, help="Server name for launching the app. 127.0.0.1 for localhost; 0.0.0.0 to allow access from other machines in the local network. Might also give access to external users depends on the firewall settings.")

    os.environ["USER"] = os.getenv("USER", "user")
    args = parser.parse_args()
    DEMO_PATH = args.demo_path
    TMP_PATH = args.tmp_path
    MODELS_PATH = args.models_path

    app = get_app()
    app.queue().launch(share=args.share, server_name=args.server_name, server_port=args.port)
