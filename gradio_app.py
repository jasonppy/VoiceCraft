import gradio as gr
import torch
import torchaudio
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
from models import voicecraft
import os
import io


def load_models(whisper_model_choice, voicecraft_model_choice):
    whisper_model, voicecraft_model = None, None
    if whisper_model_choice is not None:
        import whisper
        from whisper.tokenizer import get_tokenizer
        whisper_model = {
            "model": whisper.load_model(whisper_model_choice),
            "tokenizer": get_tokenizer(multilingual=False)
        }

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    voicecraft_name = f"{voicecraft_model_choice}.pth"
    ckpt_fn = f"./pretrained_models/{voicecraft_name}"
    encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
    if not os.path.exists(ckpt_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/{voicecraft_name}\?download\=true")
        os.system(f"mv {voicecraft_name}\?download\=true ./pretrained_models/{voicecraft_name}")
    if not os.path.exists(encodec_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
        os.system(f"mv encodec_4cb2048_giga.th ./pretrained_models/encodec_4cb2048_giga.th")

    ckpt = torch.load(ckpt_fn, map_location="cpu")
    model = voicecraft.VoiceCraft(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    voicecraft_model = {
        "ckpt": ckpt,
        "model": model,
        "text_tokenizer": TextTokenizer(backend="espeak"),
        "audio_tokenizer": AudioTokenizer(signature=encodec_fn)
    }

    return [
        whisper_model,
        voicecraft_model,
        gr.Audio(interactive=True),
    ]


def transcribe(whisper_model, audio_path):
    if whisper_model is None:
        raise gr.Error("Whisper model not loaded")
    
    number_tokens = [
        i
        for i in range(whisper_model["tokenizer"].eot)
        if all(c in "0123456789" for c in whisper_model["tokenizer"].decode([i]).removeprefix(" "))
    ]
    result = whisper_model["model"].transcribe(audio_path, suppress_tokens=[-1] + number_tokens, word_timestamps=True)
    words = [word_info for segment in result["segments"] for word_info in segment["words"]]
    
    transcript = result["text"]
    transcript_with_start_time = " ".join([f"{word['start']} {word['word']}" for word in words])
    transcript_with_end_time = " ".join([f"{word['word']} {word['end']}" for word in words])

    choices = [f"{word['start']} {word['word']} {word['end']}" for word in words]

    return [
        transcript, transcript_with_start_time, transcript_with_end_time,
        gr.Dropdown(value=choices[0], choices=choices, interactive=True), # edit_from_word
        gr.Dropdown(value=choices[-1], choices=choices, interactive=True), # edit_to_word
        words
    ]


def get_output_audio(audio_tensors, codec_audio_sr):
    result = torch.cat(audio_tensors, 1)
    buffer = io.BytesIO()
    torchaudio.save(buffer, result, int(codec_audio_sr), format="wav")
    buffer.seek(0)
    return buffer.read()


def run(voicecraft_model, left_margin, right_margin, codec_audio_sr, codec_sr, top_k, top_p, temperature,
        stop_repetition, sample_batch_size, kvcache, silence_tokens,
        audio_path, word_info, transcript, smart_transcript,
        mode, prompt_end_time, edit_start_time, edit_end_time,
        split_text, selected_sentence, previous_audio_tensors):
    if voicecraft_model is None:
        raise gr.Error("VoiceCraft model not loaded")
    if smart_transcript and (word_info is None):
        raise gr.Error("Can't use smart transcript: whisper transcript not found")

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
                for word in word_info:
                    if word["end"] < prompt_end_time:
                        target_transcript += word["word"]
                    elif (word["start"] + word["end"]) / 2 < prompt_end_time:
                        # include part of the word it it's big, but adjust prompt_end_time
                        target_transcript += word["word"]
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
                                                voicecraft_model["ckpt"]["config"],
                                                voicecraft_model["ckpt"]["phn2num"],
                                                voicecraft_model["text_tokenizer"], voicecraft_model["audio_tokenizer"],
                                                audio_path, target_transcript, device, decode_config,
                                                prompt_end_frame)
        else:
            from inference_speech_editing_scale import inference_one_sample

            if smart_transcript:
                target_transcript = ""
                for word in word_info:
                    if word["start"] < edit_start_time:
                        target_transcript += word["word"]
                    else:
                        break
                target_transcript += f" {sentence}"
                for word in word_info:
                    if word["end"] > edit_end_time:
                        target_transcript += word["word"]
            else:
                target_transcript = sentence

            inference_transcript += target_transcript + "\n"

            morphed_span = (max(edit_start_time - left_margin, 1 / codec_sr), min(edit_end_time + right_margin, audio_dur))
            mask_interval = [[round(morphed_span[0]*codec_sr), round(morphed_span[1]*codec_sr)]]
            mask_interval = torch.LongTensor(mask_interval)
            
            _, gen_audio = inference_one_sample(voicecraft_model["model"],
                                                voicecraft_model["ckpt"]["config"],
                                                voicecraft_model["ckpt"]["phn2num"],
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
    info = torchaudio.info(audio_path)
    max_time = round(info.num_frames / info.sample_rate, 2)
    return [
        gr.Slider(maximum=max_time, value=max_time),
        gr.Slider(maximum=max_time, value=0),
        gr.Slider(maximum=max_time, value=max_time),
    ]

    
def change_mode(mode):
    return [
        gr.Slider(visible=mode != "Edit"),
        gr.Radio(visible=mode == "Long TTS"),
        gr.Radio(visible=mode == "Edit"),
        gr.Row(visible=mode == "Edit"),
        gr.Accordion(visible=mode == "Edit"),
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

demo_text = {
    "TTS": {
        "smart": "I cannot believe that the same model can also do text to speech synthesis as well!",
        "regular": "But when I had approached so near to them, the common I cannot believe that the same model can also do text to speech synthesis as well!"
    },
    "Edit": {
        "smart": "saw the mirage of the lake in the distance,",
        "regular": "But when I saw the mirage of the lake in the distance, which the sense deceives, Lost not by distance any of its marks,"
    },
    "Long TTS": {
        "smart": "You can run TTS on a big text!\n"
                 "Just write it line-by-line. Or sentence-by-sentence.\n"
                 "If some sentences sound odd, just rerun TTS on them, no need to generate the whole text again!",
        "regular": "But when I had approached so near to them, the common You can run TTS on a big text!\n"
                   "But when I had approached so near to them, the common Just write it line-by-line. Or sentence-by-sentence.\n"
                   "But when I had approached so near to them, the common If some sentences sound odd, just rerun TTS on them, no need to generate the whole text again!"
    }
}

all_demo_texts = {vv for k, v in demo_text.items() for kk, vv in v.items()}

demo_words = [
    '0.0  But 0.12', '0.12  when 0.26', '0.26  I 0.44', '0.44  had 0.6', '0.6  approached 0.94', '0.94  so 1.42',
    '1.42  near 1.78', '1.78  to 2.02', '2.02  them, 2.24', '2.52  the 2.58', '2.58  common 2.9', '2.9  object, 3.3', 
    '3.72  which 3.78', '3.78  the 3.98', '3.98  sense 4.18', '4.18  deceives, 4.88', '5.06  lost 5.26', '5.26  not 5.74',
    '5.74  by 6.08', '6.08  distance 6.36', '6.36  any 6.92', '6.92  of 7.12', '7.12  its 7.26', '7.26  marks. 7.54'
]


def update_demo(mode, smart_transcript, edit_word_mode, transcript, edit_from_word, edit_to_word, prompt_end_time):
    if transcript not in all_demo_texts:
        return transcript, edit_from_word, edit_to_word, prompt_end_time
    
    replace_half = edit_word_mode == "Replace half"
    return [
        demo_text[mode]["smart" if smart_transcript else "regular"],
        "0.26  I 0.44" if replace_half else "0.44  had 0.6",
        "3.72  which 3.78" if replace_half else "2.9  object, 3.3",
        3.01,
    ]


with gr.Blocks() as app:
    with gr.Row():
        with gr.Column(scale=2):
            load_models_btn = gr.Button(value="Load models")
        with gr.Column(scale=5):
            with gr.Accordion("Select models", open=False):
                with gr.Row():
                    voicecraft_model_choice = gr.Radio(label="VoiceCraft model", value="giga830M", choices=["giga330M", "giga830M"])
                    whisper_model_choice = gr.Radio(label="Whisper model", value="base.en",
                                                    choices=[None, "tiny.en", "base.en", "small.en", "medium.en", "large"])

    with gr.Row():
        with gr.Column(scale=2):
            input_audio = gr.Audio(value="./demo/84_121550_000074_000000.wav", label="Input Audio", type="filepath", interactive=False)
            with gr.Group():
                original_transcript = gr.Textbox(label="Original transcript", lines=5, interactive=False,
                                                 info="Use whisper model to get the transcript. Fix it if necessary.")
                with gr.Accordion("Word start time", open=False):
                    transcript_with_start_time = gr.Textbox(label="Start time", lines=5, interactive=False, info="Start time before each word")
                with gr.Accordion("Word end time", open=False):
                    transcript_with_end_time = gr.Textbox(label="End time", lines=5, interactive=False, info="End time after each word")

                transcribe_btn = gr.Button(value="Transcribe")
            
        with gr.Column(scale=3):
            with gr.Group():
                transcript = gr.Textbox(label="Text", lines=7, value=demo_text["TTS"]["smart"])
                with gr.Row():
                    smart_transcript = gr.Checkbox(label="Smart transcript", value=True)
                    with gr.Accordion(label="?", open=False):
                        info = gr.HTML(value=smart_transcript_info)
                mode = gr.Radio(label="Mode", choices=["TTS", "Edit", "Long TTS"], value="TTS")
                
                prompt_end_time = gr.Slider(label="Prompt end time", minimum=0, maximum=7.93, step=0.01, value=3.01)
                split_text = gr.Radio(label="Split text", choices=["Newline", "Sentence"], value="Newline", visible=False,
                                      info="Split text into parts and run TTS for each part.")
                edit_word_mode = gr.Radio(label="Edit word mode", choices=["Replace half", "Replace all"], value="Replace half", visible=False,
                                          info="What to do with first and last word")
                with gr.Row(visible=False) as segment_control:
                    edit_from_word = gr.Dropdown(label="First word to edit", choices=demo_words, interactive=True)
                    edit_to_word = gr.Dropdown(label="Last word to edit", choices=demo_words, interactive=True)
                with gr.Accordion("Precise segment control", open=False, visible=False) as precise_segment_control:
                    edit_start_time = gr.Slider(label="Edit from time", minimum=0, maximum=60, step=0.01, value=0)
                    edit_end_time = gr.Slider(label="Edit to time", minimum=0, maximum=60, step=0.01, value=60)

                run_btn = gr.Button(value="Run")

        with gr.Column(scale=2):
            output_audio = gr.Audio(label="Output Audio")
            with gr.Accordion("Inference transcript", open=False):
                inference_transcript = gr.Textbox(label="Inference transcript", lines=5, interactive=False,
                                                  info="Inference was performed on this transcript.")
            with gr.Group(visible=False) as long_tts_controls:
                sentence_selector = gr.Dropdown(label="Sentence", value=None,
                                                info="Select sentence you want to regenerate")
                sentence_audio = gr.Audio(label="Sentence Audio", scale=2)
                rerun_btn = gr.Button(value="Rerun")

    with gr.Row():
        with gr.Accordion("VoiceCraft config", open=False):
            left_margin = gr.Number(label="left_margin", value=0.08)
            right_margin = gr.Number(label="right_margin", value=0.08)
            codec_audio_sr = gr.Number(label="codec_audio_sr", value=16000)
            codec_sr = gr.Number(label="codec_sr", value=50)
            top_k = gr.Number(label="top_k", value=0)
            top_p = gr.Number(label="top_p", value=0.8)
            temperature = gr.Number(label="temperature", value=1)
            stop_repetition = gr.Radio(label="stop_repetition", choices=[-1, 1, 2, 3], value=3,
                                       info="if there are long silence in the generated audio, reduce the stop_repetition to 3, 2 or even 1, -1 = disabled")
            sample_batch_size = gr.Number(label="sample_batch_size", value=4, precision=0,
                                          info="generate this many samples and choose the shortest one")
            kvcache = gr.Radio(label="kvcache", choices=[0, 1], value=1,
                                info="set to 0 to use less VRAM, but with slower inference")
            silence_tokens = gr.Textbox(label="silence tokens", value="[1388,1898,131]")

    
    whisper_model = gr.State()
    voicecraft_model = gr.State()
    audio_tensors = gr.State()
    word_info = gr.State()

    
    mode.change(fn=update_demo,
                inputs=[mode, smart_transcript, edit_word_mode, transcript, edit_from_word, edit_to_word, prompt_end_time],
                outputs=[transcript, edit_from_word, edit_to_word, prompt_end_time])
    edit_word_mode.change(fn=update_demo,
                          inputs=[mode, smart_transcript, edit_word_mode, transcript, edit_from_word, edit_to_word, prompt_end_time],
                          outputs=[transcript, edit_from_word, edit_to_word, prompt_end_time])
    smart_transcript.change(fn=update_demo,
                            inputs=[mode, smart_transcript, edit_word_mode, transcript, edit_from_word, edit_to_word, prompt_end_time],
                            outputs=[transcript, edit_from_word, edit_to_word, prompt_end_time])
    
    load_models_btn.click(fn=load_models,
                          inputs=[whisper_model_choice, voicecraft_model_choice],
                          outputs=[whisper_model, voicecraft_model, input_audio])
    
    input_audio.change(fn=update_input_audio,
                       inputs=[input_audio],
                       outputs=[prompt_end_time, edit_start_time, edit_end_time])
    transcribe_btn.click(fn=transcribe,
                         inputs=[whisper_model, input_audio],
                         outputs=[original_transcript, transcript_with_start_time, transcript_with_end_time, edit_from_word, edit_to_word, word_info])

    mode.change(fn=change_mode,
                inputs=[mode],
                outputs=[prompt_end_time, split_text, edit_word_mode, segment_control, precise_segment_control, long_tts_controls])

    run_btn.click(fn=run,
                  inputs=[
                      voicecraft_model, left_margin, right_margin,
                      codec_audio_sr, codec_sr,
                      top_k, top_p, temperature,
                      stop_repetition, sample_batch_size,
                      kvcache, silence_tokens,
                      input_audio, word_info, transcript, smart_transcript,
                      mode, prompt_end_time, edit_start_time, edit_end_time,
                      split_text, sentence_selector, audio_tensors
                  ],
                  outputs=[output_audio, inference_transcript, sentence_selector, audio_tensors])
    
    sentence_selector.change(fn=load_sentence,
                             inputs=[sentence_selector, codec_audio_sr, audio_tensors],
                             outputs=[sentence_audio])
    rerun_btn.click(fn=run,
                    inputs=[
                        voicecraft_model, left_margin, right_margin,
                        codec_audio_sr, codec_sr,
                        top_k, top_p, temperature,
                        stop_repetition, sample_batch_size,
                        kvcache, silence_tokens,
                        input_audio, word_info, transcript, smart_transcript,
                        gr.State(value="Rerun"), prompt_end_time, edit_start_time, edit_end_time,
                        split_text, sentence_selector, audio_tensors
                    ],
                    outputs=[output_audio, inference_transcript, sentence_audio, audio_tensors])
    
    edit_from_word.change(fn=update_bound_word,
                          inputs=[gr.State(True), edit_from_word, edit_word_mode],
                          outputs=[edit_start_time])
    edit_to_word.change(fn=update_bound_word,
                        inputs=[gr.State(False), edit_to_word, edit_word_mode],
                        outputs=[edit_end_time])
    edit_word_mode.change(fn=update_bound_words,
                          inputs=[edit_from_word, edit_to_word, edit_word_mode],
                          outputs=[edit_start_time, edit_end_time])


if __name__ == "__main__":
    app.launch()