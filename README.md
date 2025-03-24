# talk-llama-fast

based on talk-llama https://github.com/ggerganov/whisper.cpp

Видео-инструкция на русском (Russian guide, English subs): https://youtu.be/0MEZ84uH4-E

English demo video, v0.1.3: https://www.youtube.com/watch?v=ORDfSG4ltD4

Видео на русском, v0.1.0: https://youtu.be/ciyEsZpzbM8


ТГ: https://t.me/tensorbanana

## I added:
- XTTSv2 support
- UTF8 and Russian
- Speed-ups: streaming for generation, streaming for xtts, aggresive VAD
- voice commands: Google, stop, regenerate, delete, reset, call
- generation/tts interruption when user is speaking
- wav2lip streaming

## I used: 
- whisper.cpp ggml-medium-q5_0.bin
- mistral-7b-instruct-v0.2.Q5_0.gguf
- XTTSv2 server in streaming-mode
- langchain google-serper
- wav2lip

## News
- [2025.03.24] v0.2.1. Support for `gemma-3` and `mistral-small-2503`, text inference. Image input is coming next! Synced codebase to recent llama.cpp.
- [2024.07.21] v0.2.0. Support for `gemma-2` and `mistral-nemo`. Added multiple gpu (`--main-gpu 0`, `--split-mode none`, `--tensor-split 0.5,0.5`). Added `--instruct-preset gemma`. More details in [release](https://github.com/Mozer/talk-llama-fast/releases/tag/0.2.0).
- [2024.07.21] update of upstream llama.cpp/whisper up to this [commit](https://github.com/ggerganov/whisper.cpp/commit/d207c6882247984689091ae9d780d2e51eab1df7)
- [2024.06.26] v0.1.8. Added `--min_p 0.00` sampler param. I recommend to set it 0.10 for Russian.
- [2024.05.25] Created telegram bot with multiple characters: https://t.me/talkllama And added donation page: https://github.com/Mozer/donate
- [2024.05.17] Added `talk-llama-fast-v0.1.7_no_avx2.zip` for old CPUs without AVX2 instructions (e.g. Intel i5-2500K). Use it if main version crashes without an error.
- [2024.05.09] v0.1.7. Added `--push-to-talk` option: hold "Alt" key to speak (useful with loudspeakers without headphones). And now you can use Cyrillic letters in bat files. Save them using Cyrillic "OEM 866" encoding, notepad++ supports it.
- [2024.04.30] v0.1.6. Big fix: start prompt was not working correctly.
- [2024.04.25] v0.1.5. Added keyboard input, hotkeys.
- [2024.04.17] v0.1.4. Added `--batch-size` (llama takes 0.6 GB less VRAM then it was before!), `--verbose` (to show speed). Start prompt is now not limited in length. But keep it < ctx_size for speed.
- [2024.04.04] v0.1.0. Added streaming wav2lip. With super low latency: from user speech to video it's just 1.5 seconds! Had to rewrite sillyTavern-extras, wav2lip, xtts-api-server, tts (all forked to my github). Streaming wav2lip can be used in SillyTavern. Setup guide and video are coming in a next few days. 
- [2024.03.09] v0.0.4. New params: `--stop-words` (list for llama separated by semicolon: `;`), `--min-tokens` (min tokens to output), `--split-after` (split first sentence after N tokens for xtts), `--seqrep` (detect loops: 20 symbols in 300 last symbols), `--xtts-intro` (echo random Umm/Well/...  to xtts right after user input). See [0.0.4](https://github.com/Mozer/talk-llama-fast/releases/tag/0.0.4) release for details.
- [2024.02.28] v0.0.3 `--multi-chars` param to enable different voice for each character, each one will be sent to xtts, so make sure that you have corresponding .wav files (e.g. alisa.wav). Use with voice command `Call NAME`. Video, in Russian: https://youtu.be/JOoVdHZNCcE or https://t.me/tensorbanana/876
- `--translate` param for live en_ru translation. Russian user voice is translated ru->en using whisper. Then Llama output is translated en->ru using the same mistral model, inside the same context, without any speed dropouts, no extra vram is needed. This trick gives more reasoning skills to llama in Russian, but instead gives more grammar mistakes. And more text can fit in the context, because it is stored in English, while the translation is deleted from context right after generation of each sentence. `--allow-newline` param. By default, without it llama will stop generation if it finds a new line symbol.
- [2024.02.25] I added `--vad-start-thold` param for tuning stop on speech detection (default: 0.000270; 0 to turn off). VAD checks current noise level, if it is loud - xtts and llama stops. Turn it up if you are in a noisy room, also check `--print-energy`.
- [2024.02.22] initial public release

## Notes
- llama.cpp context shifting is working great by default. I used 2048 ctx and tested dialog up to 10000 tokens - the model is still sane, no severe loops or serious problems. Llama remembers everything from a start prompt and from the last 2048 of context, but everything in the middle - is lost. No extra VRAM is used, you can have almost an endless talk without speed dropout.
- default settings are tuned for extreme low latency. If llama is interrupting you: set `--vad-last-ms 500` instead of 200 ms. If you don't like a little pause after xtts first words set `--split-after 0` instead of 5 - it will turn off first sentence splitting but it will be a little slower for the first sentence to be vocalized. 
- wav2lip is trained on small videos - recommended: 300x400 25 fps 1 minute long. Big resolution vids can cause vram out of memory errors.
- wav2lip is not trained for anime, lips will look like human, and some faces are not detected at all.
- If wav2lip often skips 2nd+ parts of video while audio is playing fine, in xtts-wav2lip.bat try changing to `--wav-chunk-sizes 20,40,100,200,300,400,9999` or even 100,200,300,400,9999 to make wav splitting less aggressive. You can also tune +- `--sleep-before-xtts 1000` in talk-llama-wav2lip.bat - it is the sleep time in ms for llama after sending each xtts request.
- in xtts_wav2lip.bat don't set `--extras-url` as http://localhost:5100/, put it as `http://127.0.0.1:5100/`. localhost option was slower by 2 seconds in my case, not sure why.
- if you are using bluetooth headphones and audio is lagging after video you can tune this lag: in `SillyTavern-extras\modules\wav2lip\server_wav2lip.py` in play_video_with_audio at line 367 set `sync_audio_delta_bytes = 5000`.
- wav2lip video is played on the same device as the host. Currently it can't be run on a remote server like google colab. Mobile phones are also not supported ATM.
- wav2lip can be used with original SillyTavern (just xtts+wav2lip, no speech-to-text, no voice interruption). No extra extensions required, just follow installation process.
- VRAM usage: mistral-7B-q5_0 + whisper-medium-q5_0.bin: 7.5 GB, xtts: 2.7 GB, wav2lip: 0.8 GB = Total of 11.0 GB. If you have just 8 GB: use smaller quant of llama!; try using --lowvram with xtts or even start xtts on cpu instead of gpu (`-d=cpu` but it is slow). Try to turn off streaming in xtts: set streaming chunk size as a single number in xtts_wav2lip.bat (--wav-chunk-sizes 9999). It will be slower, but less overhead for multiple small requests.
- to use with speakers (not headphones):
1. You can turn off the interruption of bot speech by noise `--vad_start_thold 0`.
2. Optional: there is a command for “awakening” `--wake-command "Anna,"` (a comma after the name is required). Now, only those phrases that begin with the name "Anna" will go into the chat on your behalf. This will partially help when working with speakers or in a noisy room.

## Languages
Whisper STT supported languages: Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh.

XTTSv2 supported languages: English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko), Hindi (hi).

Mistral officially supports: English, French, Italian, German, Spanish. But it can also speak some other languages, but not so fluent (e.g. Russian is not officially supported, but it is there).

## Requirements
- Windows 10/11 x64
- python, cuda
- 16 GB RAM
- Recommended: nvidia GPU with 12 GB vram. Minimum: nvidia with 6 GB. For 6GB or 8GB vram see [tweaks](https://github.com/Mozer/talk-llama-fast?tab=readme-ov-file#tweaks-for-6-and-8-gb-vram)
- For AMD, macos, linux, android - first you need to compile everything. I don't know if it works. 
- Android version is TODO.

## Installation
### For Windows 10/11 x64 with CUDA.
- Check that you have Cuda Toolkit 11.x. If not - install: https://developer.nvidia.com/cuda-11-8-0-download-archive
- Download latest [release](https://github.com/Mozer/talk-llama-fast/releases) in zip. Extract it's contents.
- Download whisper medium model to folder with talk-llama.exe: [for English](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-medium.en-q5_0.bin) or [for Russian](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-medium-q5_0.bin) (or even [large-v3-q4_0.bin](https://huggingface.co/Ftfyhh/whisper-ggml-q4_0-models/blob/main/whisper-ggml-large-v3-q4_0.bin) it is larger but much better for Russian). You can try small-q5 if you don't have much VRAM. For English try [distilled medium](https://huggingface.co/distil-whisper/distil-medium.en/blob/main/ggml-medium-32-2.en.bin), it takes 100 MB less VRAM.
- Download LLM to same folder [mistral-7b-instruct-v0.2.Q5_0](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q5_0.gguf) or [Vikhr-7B-0.4 for Russian](https://huggingface.co/Vikhrmodels/Vikhr-7B-instruct_0.4-GGUF/resolve/main/vikhr-7b-instruct_0.4.Q5_0.gguf). You can try q4_K_S or q3 if you don't have much VRAM.
- Now let's install my modified sillyTavern-extras, wav2lip, xtts-api-server, tts (all from my github). Note: xtts-api-server conflicts with SillyTavern-Extras (xtts deepspeed needs torch 2.1 but some package in extras (torchvision 0.17.2) needs torch 2.2). Before that i was able to run them both in 3.11, but users reported several problems trying to install xtts-api-server together with SillyTavern-Extras without conda. So now we will install everything with 2 different conda environments with different torches (7 GB for each conda, i know it is big). It has 2 parts: for xtts and for SillyTavern-Extras. If you know how to install everything in 1 conda environment step by step - open a PR.

install [miniconda](https://docs.anaconda.com/free/miniconda/). During installation make sure to check "Add Miniconda3 to my PATH environment variable" - it's important.

Open \xtts\ folder where you extracted talk-llama-fast-v0.1.3.zip. In this folder open a `cmd` and run line by line:
```
conda create -n xtts
conda activate xtts
conda install python=3.11
conda install git

pip install git+https://github.com/Mozer/xtts-api-server pydub
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/Mozer/tts
conda deactivate
```
- if there are some errors with xtts-api-server installation, check manuals (not my xtts, they install original xtts, not modified): [xtts-api-server](https://github.com/daswer123/xtts-api-server?tab=readme-ov-file#installation) or another manual (didn't work for me) [from sillytavern](https://docs.sillytavern.app/extras/extensions/xtts/) or this [stackoverflow TTS install manual](https://stackoverflow.com/questions/66726331/how-can-i-run-mozilla-tts-coqui-tts-training-with-cuda-on-a-windows-system), but it uses some different versions.
I remember that when I first installed xtts-api-server it asked to install some full version of [visual-cpp-build-tools](https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/). The default download page from MS wasn't working for me, so i had to google and found it elsewher. [VS_BuildTools.exe screenshot 1](https://github.com/Mozer/talk-llama-fast/assets/1599013/23627998-28f7-4eeb-9bc5-be54c1a68217), [screenshot 2](https://github.com/Mozer/talk-llama-fast/assets/1599013/b7ff8401-c5b3-4f5c-abdb-e527f296b12d). Or maybe it was [VC_redist.x86.exe](https://learn.microsoft.com/ru-ru/cpp/windows/latest-supported-vc-redist?view=msvc-170).

- In the same dir where you are now, create second conda for extras
```
conda create -n extras
conda activate extras
conda install python=3.11
conda install git

git clone https://github.com/Mozer/SillyTavern-Extras
cd SillyTavern-extras
pip install -r requirements.txt
cd modules
git clone https://github.com/Mozer/wav2lip
cd wav2lip
pip install -r requirements.txt
conda deactivate
```

- Notice: that \wav2lip\ was installed inside \SillyTavern-extras\modules\ folder. That's important.
- Edit xtts_wav2lip.bat, change `--output` from c:\\DATA\\LLM\\SillyTavern-Extras\\tts_out\\ to actual path where your \\SillyTavern-Extras\\tts_out\\ dir is located. Don't forget the trailing slashes here.
- Optional: if you have just 6 or 8 GB of vram - in talk-llama-wav2lip.bat find and change to `-ngl 0`. It will move mistral from GPU to CPU+RAM.
- Optional: edit talk-llama-wav2lip.bat or talk-llama-wav2lip-ru.bat, make sure it has correct LLM and whisper model names that you downloaded. (Full params description is below).
- Download [ffmpeg full](https://www.gyan.dev/ffmpeg/builds/), put into your PATH environment (how to: https://phoenixnap.com/kb/ffmpeg-windows). Then download h264 codec .dll of required version from https://github.com/cisco/openh264/releases and put to /system32 or /ffmpeg/bin dir. In my case for Windows 11 it was openh264-1.8.0-win64.dll. Wav2lip will work without this dll but will print an error.


## Running
- In /SillyTavern-extras/ double click `silly_extras.bat`. Wait until it downloads wav2lip checkpoint and makes face detection for new video if needed.
- In /xtts/ double click `xtts_wav2lip.bat` to start xtts server with wav2lip video. OR run xtts_streaming_audio.bat to start xtts server with audio without video. NOTE: On the first run xtts will download DeepSpeed from github. If deepspeed fails to download "Warning: Retyring (Retry... ReadTimoutError...") - turn on VPN to download deepspeed (27MB) and xtts checkpoint (1.8GB), then you can turn it off). Xtts checkpoint can be downloaded without VPN. But if you interrupt download - checkpoint will be broken - you have to manually delete \xtts_models\ dir and restart xtts. 
- Double click `talk-llama-wav2lip.bat` or `talk-llama-wav2lip-ru.bat` or talk-llama-just-audio.bat. Don't run exe, just bat. NOTE: if you have cyrillic (Russian) letters in .bat - save it in Cyrillic "OEM 866" encoding (notepadd++ supports it).
- Start speaking. 

### Tweaks for 6 and 8 GB vram
- use CPU instead of GPU, it will be a bit slower (5-6 s): in talk-llama-wav2lip.bat find and change ngl to `-ngl 0` (mistral has 33 layers, try values from 0 to 33 to find best speed)
- set smaller context for llama: `--ctx_size 512`
- set `--lowvram` in xtts_wav2lip.bat, that will move xtts model from GPU to RAM after each xtts request (but it will be slower)
- set `--wav-chunk-sizes=9999` in xtts_wav2lip.bat it will be a bit slower, but will have less wav2lip requests.
- try smaller whisper mode, for example [small](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-small-q5_1.bin) or [english distilled medium](https://huggingface.co/distil-whisper/distil-medium.en/blob/main/ggml-medium-32-2.en.bin)

### Optional
- Put new xtts voices into `\xtts\speakers\`. I recommend  16 bit mono, 22050Hz 10 seconds long wav without noises and music. Use audacity to edit.
- Put new videos into `\SillyTavern-extras\modules\wav2lip\input\`. I recommend 300x400 25 fps 1 minute long, don't put high res vids, they use A LOT of vram. One video into one folder. Folder name should be the same as desired xtts voice name and a char name in talk-llama-wav2lip.bat. E.g. Anna.wav and \Anna\youtube_ann_300x400.mp4 for character with the name Anna. With `--multi-chars` param talk-llama will pass name of the new character to xtts and wav2lip even if this character is not defined in bat or start prompt. If xtts won't find that voice it will use default voice. If wav2lip won't find that video it will use default video.
- Put character descrition and some replies to assistant.txt. 
- Use exact same name for your character and for .wav file and for video folder name. You can also make copies of audio/video files (e.g. Kurt Cobain and Kurt). Now you can address him both ways.
- For better Russian in XTTS check my finetune: https://huggingface.co/Ftfyhh/xttsv2_banana But it is not for streaming (hallucinates at short replies). Use with default xtts in silly tavern.

#### Optional, better coma handling for xtts - only for xtts audio without wav2lip video
Better speech, but a little slower for first sentence. Xtts won't split sentences by coma ',':
c:\Users\[USERNAME]\miniconda3\Lib\site-packages\stream2sentence\stream2sentence.py
line 191, replace 
```sentence_delimiters = '.?!;:,\n…)]}。'```
with
```sentence_delimiters = '.?!;:\n…)]}。'```

#### Optional, google search plugin
- download [search_server.py](https://github.com/Mozer/talk-llama-fast/blob/master/search_server.py) from my repo
- install langchain: `pip install langchain`
- sign up at https://serper.dev/api-key it is free and fast, it will give you 2500 free searches. Get an API key, paste it to search_server.py at line 13 `os.environ["SERPER_API_KEY"] = "your_key"`
- start search server by double clicking search_server.py. Now you can use voice commands like these: `Please google who is Barack Obama` or `Пожалуйста погугли погоду в Москве`.


## Building, optional
- for nvidia and Windows. Other systems - try yourself.
- download https://www.libsdl.org/release/SDL2-devel-2.28.5-VC.zip extract to /whisper.cpp/SDL2/ folder
- install libcurl using vcpkg:
```
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
vcpkg install curl[tool]
```
- Modify path `c:\\DATA\\Soft\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake` below to folder where you installed vcpkg. Then build.
```
git clone https://github.com/Mozer/talk-llama-fast
cd talk-llama-fast
set SDL2_DIR=SDL2\cmake
cmake.exe -DWHISPER_SDL2=ON -DWHISPER_CUBLAS=0 -DGGML_CUDA=1 -DCMAKE_TOOLCHAIN_FILE="c:\\DATA\\Soft\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake" -B build -DCMAKE_BUILD_PARALLEL_LEVEL=4
cmake.exe --build build --config release --target clean
del build\bin\Release\talk-llama.exe & cmake.exe --build build --config release --parallel 4

for old CPU's without AVX2: cmake.exe -DWHISPER_NO_AVX2=1 -DWHISPER_SDL2=ON -DWHISPER_CUBLAS=0 -DGGML_CUDA=1 -DCMAKE_TOOLCHAIN_FILE="c:\\DATA\\Soft\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake" -B build -DCMAKE_BUILD_PARALLEL_LEVEL=4
then same 2 last lines

compilation can take 30+ minutes.
```


## talk-llama.exe params
```
  -h,       --help           [default] show this help message and exit
  -t N,     --threads N      [4      ] number of threads to use during computation
  -vms N,   --voice-ms N     [10000  ] voice duration in milliseconds
  -c ID,    --capture ID     [-1     ] capture device ID
  -mt N,    --max-tokens N   [32     ] maximum number of tokens per audio chunk
  -ac N,    --audio-ctx N    [0      ] audio context size (0 - all)
  -ngl N,   --n-gpu-layers N [999    ] number of layers to store in VRAM
  -vth N,   --vad-thold N    [0.60   ] voice avg activity detection threshold
  -vths N,  --vad-start-thold N [0.000270] vad min level to stop tts, 0: off, 0.000270: default
  -vlm N,   --vad-last-ms N  [0      ] vad min silence after speech, ms
  -fth N,   --freq-thold N   [100.00 ] high-pass frequency cutoff
  -su,      --speed-up       [false  ] speed up audio by x2 (not working)
  -tr,      --translate      [false  ] translate from source language to english
  -ps,      --print-special  [false  ] print special tokens
  -pe,      --print-energy   [false  ] print sound energy (for debugging)
  --debug                    [false  ] print debug info
  -vp,      --verbose-prompt [false  ] print prompt at start
  --verbose                  [false  ] print speed
  -ng,      --no-gpu         [false  ] disable GPU
  -p NAME,  --person NAME    [Georgi ] person name (for prompt selection)
  -bn NAME, --bot-name NAME  [LLaMA  ] bot name (to display)
  -w TEXT,  --wake-command T [       ] wake-up command to listen for
  -ho TEXT, --heard-ok TEXT  [       ] said by TTS before generating reply
  -l LANG,  --language LANG  [en     ] spoken language
  -mw FILE, --model-whisper  [models/ggml-base.en.bin] whisper model file
  -ml FILE, --model-llama    [models/ggml-llama-7B.bin] llama model file
  -s FILE,  --speak TEXT     [./examples/talk-llama/speak] command for TTS
  --prompt-file FNAME        [       ] file with custom prompt to start dialog
  --instruct-preset TEXT     [       ] instruct preset to use without .json
  --session FNAME                   file to cache model state in (may be large!) (default: none)
  -f FNAME, --file FNAME     [       ] text output file name
   --ctx_size N              [2048   ] Size of the prompt context
  -b N,     --batch-size N   [64     ] Size of input batch size
  -n N,     --n_predict N    [64     ] Max number of tokens to predict
  --temp N                   [0.90   ] Temperature
  --top_k N                  [40.00  ] top_k
  --top_p N                  [1.00   ] top_p
  --min_p N                  [0.00   ] min_p
  --repeat_penalty N         [1.10   ] repeat_penalty
  --repeat_last_n N          [256    ] repeat_last_n
  --main-gpu N               [0      ] main GPU id, starting from 0
  --split-mode NAME          [none   ] GPU split mode: 'none' or 'layer'
  --tensor-split NAME        [(null) ] Tensor split, list of floats: 0.5,0.5
  --xtts-voice NAME          [emma_1 ] xtts voice without .wav
  --xtts-url TEXT            [http://localhost:8020/] xtts/silero server URL, with trailing slash
  --xtts-control-path FNAME  [c:\DATA\LLM\xtts\xtts_play_allowed.txt] not used anymore
  --xtts-intro               [false  ] xtts instant short random intro like Hmmm.
  --sleep-before-xtts        [0      ] sleep llama inference before xtts, ms.
  --google-url TEXT          [http://localhost:8003/] langchain google-serper server URL, with /
  --allow-newline            [false  ] allow new line in llama output
  --multi-chars              [false  ] xtts will use same wav name as in llama output
  --push-to-talk             [false  ] hold Alt to speak
  --seqrep                   [false  ] sequence repetition penalty, search last 20 in 300
  --split-after N            [0      ] split after first n tokens for tts
  --min-tokens N             [0      ] min new tokens to output
  --stop-words TEXT          [       ] llama stop w: separated by ;
```

## Voice commands:
Full list of commands and variations is in `talk-llama.cpp`, search `user_command`.
- Stop (остановись, Ctrl+Space)
- Regenerate (переделай, , Ctrl+Right) - will regenerate llama answer
- Delete (удали, Ctrl+Delete) - will delete user question and llama answer.
- Delete 3 messages (удали 3 сообщениия)
- Reset (удали все, Ctrl+R) - will delete all context except for a initial prompt
- Google something (погугли что-то)
- Сall NAME (позови Алису)

## Known bugs
- if you have missing cuda .dll errors - see this [issue](https://github.com/Mozer/talk-llama-fast/issues/5)
- if whisper doesn't hear your voice - see this [issue](https://github.com/Mozer/talk-llama-fast/issues/5)
- Rope context - is not implemented. Use context shifting (enabled by default).
- sometimes whisper is hallucinating, need to put hallucinations into stop-words. Check `misheard text` in `talk-llama.cpp`
- don't put cyrillic (русские) letters for characters or paths in .bat files, they may not work nice because of weird encoding. Copy text from .bat, paste into `cmd` if you need to use cyrillic letters with talk-llama-fast.exe.
- During first run wav2lip will run face detection with a newly added video. It will take about 30-60 s, but it happens just once and then it is saved to cache. And there is a bug with face detection wich slows down everything (memory leak). You need to restart Silly Tavern Extras after face detection is finished.
- Sometimes wav2lip video window disappears but audio is still playing fine. If the video window doesn't come back automatically - restart Silly Tavern Extras.
- if you restart xtts you need to restart silly-tavern-extras. Otherwise wav2lip will start playing wrong segments of already created videos.
- Sometimes when you type fast the first letter of you message is not printed.

## Licenses
- talk-llama-fast - MIT License - OK for commercial use
- whisper.cpp - MIT License - OK for commercial use
- whisper - MIT License - OK for commercial use
- TTS(xtts) - Mozilla Public License 2.0 - OK for commercial use
- xtts-api-server - MIT License - OK for commercial use
- Silly Extras - GNU Public License v3.0 - OK for commercial use
- Mistral 7B - Apache 2.0 license - OK for commercial use
- Wav2Lip - non-commercial - for commercial requests, please contact synclabs.so directly

## Contacts
Reddit: https://www.reddit.com/user/tensorbanana2

ТГ: https://t.me/tensorbanana

Donate: https://github.com/Mozer/donate