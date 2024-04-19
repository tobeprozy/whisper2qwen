from bmwhisper.transcribe import *
from bmwhisper import available_models
from bmwhisper import load_model


from qwen import Qwen


def argsparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--bmodel_dir", type=str, default="../models/BM1684X/", help="the path of bmodels; uses ../models/BM1684X/ by default")

    parser.add_argument('--dev_id', type=int, default=0, help='dev id for sophgo tpu')

    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["txt", "vtt", "srt", "tsv", "json", "all"], help="format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--word_timestamps", type=str2bool, default=False, help="(experimental) extract word-level timestamps and refine the results based on them")
    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default="\"\'.。,，!！?？:：”)]}、", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt")
    parser.add_argument("--max_line_width", type=optional_int, default=None, help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="(requires --word_timestamps True) the maximum number of lines in a segment")
    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    parser.add_argument("--padding_size", type=optional_int, default=448, help="max pre-allocation size for the key-value cache")
    parser.add_argument("--loop_profile", action="store_true", help="whether to print loop times")

    
    parser.add_argument('--qbmodel', type=str, default='../models/BM1684X/qwen-7b_int4_1dev.bmodel', help='path of bmodel')
    parser.add_argument('--qtoken', type=str, default='./token_config/', help='path of tokenizer')
    parser.add_argument('--qdev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args


def whisper2qwen(args):
    start_time = time.time()
    
    args["model_name"] = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    loop_profile = args.pop("loop_profile")
    os.makedirs(output_dir, exist_ok=True)

    model_name = args["model_name"]
    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead."
            )
        args["language"] = "en"

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)

    

    model = load_model(args)
    pop_list = ["model_name", "bmodel_dir", "dev_id"]
    for arg in pop_list:
        args.pop(arg)

    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if not args["word_timestamps"]:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} requires --word_timestamps True")
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}
    audio_list=args.pop("audio")
    for audio_path in audio_list:
        if os.path.isdir(audio_path):
            all_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path)]
            audio_list.extend(all_files)
            continue
        print()
        print("{:=^100}".format(f" Start "))
        print(f"### audio_path: {os.path.basename(audio_path)}")
        audio_start_time = time.time()
        model.init_cnt()
        model.init_time()
        result = transcribe(model, audio_path, temperature=temperature, **args)
        # writer(result, audio_path, writer_args)
        print("Question: "+result["text"])
        qwen.chat(result["text"])

        total_time = time.time() - audio_start_time
        preprocess_time = total_time - model.inference_time
        if loop_profile:
            model.print_cnt()
        print()
        print(f"Preprocess time: {preprocess_time}s")
        print(f"Inference time: {model.inference_time}s")
        print(f"Total time: {total_time}s")

    print("{:=^100}".format(f" End "))
    print("{:-^100}".format(f" {len(audio_list)} audio(s) total time: {time.time() - start_time} seconds "))

import sys

if __name__ == "__main__":
    args = argsparser()
    args = args.__dict__

    args_qwen = {"qbmodel":args["qbmodel"],"qtoken":args["qtoken"],"qdev_id":args["qdev_id"]}

    for k,v in args_qwen.items():
        args.pop(k)

    args_qwen = argparse.Namespace(**args_qwen)
    
    qwen = Qwen(args_qwen)
    whisper2qwen(args)

    print('all done')

# python3 -m pdb voice2qwen.py ../datasets/test/demo.wav --model medium --bmodel_dir ../models/BM1684X --dev_id 0  --output_dir ./result/ --output_format txt --qbmodel ../models/BM1684X/qwen-7b_int4_1dev.bmodel --qtoken token_config --qdev_id 0




