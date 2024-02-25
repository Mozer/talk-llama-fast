# talk-llama-fast

Early pre beta!

based on talk-llama https://github.com/ggerganov/whisper.cpp

## I added:
- xTTSv2 support
- UTF8 and Russian
- Speed-ups: streaming for generation, streaming for xtts, aggresive VAD
- voice commands: Google, stop, regenerate, reset
- generation/tts interruption when user is speaking

## I used: 
- whisper.cpp ggml-medium-q5_0.bin
- mistral-7b-instruct-v0.2.Q6_K.gguf
- xTTSv2 server in streaming-mode
- langchain google-serper

## News
- 2024.02.25 - I added `--vad-start-thold` param for tuning stop on speech detection (0.000270: default, 0 to turn off). VAD checks current noise level, if it is loud - xtts and llama stops. Turn it up if you are in a noisy room, also check `--print-energy`.
- 2024.02.22 - initial public release


## Requirements
- Windows 10/11 x64
- python, cuda
- nvidia 3060 12 GB vram, but i guess you can try with 8 GB. Also you can try to use CPU instead of GPU, but it will be slow (you need to build cpu version yourself).
- For AMD, macos, linux, android - first you need to compile everything. I don't know if it works. 
- Android version is TODO.

## Installation
### For Windows 10/11 x64 with CUDA
- Download anywhere all files from the latest release (Releases section is on the right)
- install https://github.com/daswer123/xtts-api-server Use this manual: https://docs.sillytavern.app/extras/extensions/xtts/
- Download /xtts directory from my repostory, keep the structure. Run xtts_streaming.bat to start xtts server.
- Download whisper model to folder with talk-llama.exe: https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-medium-q5_0.bin (for Russian) or https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-medium.en-q5_0.bin (for English). You can try small-q5 if you don't have much VRAM.
- Download LLM to same folder https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q6_K.gguf , you can try q4_K_S if you don't have much VRAM.
- Optional: edit talk-llama.bat or talk-llama_ru.bat, change params if needed (params description is below). Also check optional section below for speed-ups and google plugin.
- Click talk-llama.bat or talk-llama_ru.bat, start speaking.  

### Optional
#### stop xtts when user is speaking
- To stop playing XTTS: In talk-llama.bat change param --xtts-control-path to full path where you have xtts_play_allowed.txt
- Then you need to modify c:\Users\[USERNAME]\miniconda3\Lib\site-packages\xtts_api_server\RealtimeTTS\text_to_stream.py
- download /xtts/text_to_stream.py from my repo, compare its contents with original file (e.g. using notepad++ compare plugin), make changes. I will make automatic patcher later.

#### Optional, better coma handling for xtts
Better speech, but a little slower for first sentence:
c:\Users\[USERNAME]\miniconda3\Lib\site-packages\stream2sentence\stream2sentence.py
line 191, replace 
```
sentence_delimiters = '.?!;:,\n…)]}。'
with
sentence_delimiters = '.?!;:\n…)]}。'
```

#### Optional, google search plugin
- download search_server.py from my repo
- install langchain: `pip install langchain`
- sign up at https://serper.dev/api-key it is free and fast, it will give you 2500 free searches. Get an API key, paste it to search_server.py at line 15 `os.environ["SERPER_API_KEY"] = "your_key"`
- start search server by double clicking it. Now you can use voice commands like these: `Please google who is Barack Obama` or `Пожалуйста погугли погоду в Москве`.


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
cmake.exe -DWHISPER_SDL2=ON -DWHISPER_CUBLAS=1 -DCMAKE_TOOLCHAIN_FILE="c:\\DATA\\Soft\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake" -B build
cmake.exe --build build --config release --target clean
del build\bin\Release\talk-llama.exe & cmake.exe --build build --config release
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
  -su,      --speed-up       [false  ] speed up audio by x2 (reduced accuracy)
  -tr,      --translate      [false  ] translate from source language to english
  -ps,      --print-special  [false  ] print special tokens
  -pe,      --print-energy   [false  ] print sound energy (for debugging)
  -vp,      --verbose-prompt [false  ] print prompt at start
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
  --session FNAME                   file to cache model state in (may be large!) (default: none)
  -f FNAME, --file FNAME     [       ] text output file name
   --ctx_size N              [2048   ] Size of the prompt context
  -n N, --n_predict N        [64     ] Number of tokens to predict
  --temp N                   [0.90   ] Temperature
  --top_k N                  [40.00  ] top_k
  --top_p N                  [1.00   ] top_p
  --repeat_penalty N         [1.10   ] repeat_penalty
  --xtts-voice NAME          [emma_1 ] xtts voice without .wav
  --xtts-url TEXT            [http://localhost:8020/] xtts/silero server URL, with trailing slash
  --xtts-control-path FNAME  [c:\DATA\LLM\xtts\xtts_play_allowed.txt] path to xtts_play_allowed.txt  --google-url TEXT          [http://localhost:8003/] langchain google-serper server URL, with /
```

## Voice commands:
Full list of commands and variations is in `talk-llama.cpp`, search `user_command`.
- Stop (остановись)
- Regenerate (переделай)
- Reset (удали все)
- Google something (погугли что-то)

## Bugs
- `Reset` voice command won't work nice if  current context length is over --ctx_size
- Rope context scaling is not working nice
- sometimes whisper is hallucinating, need to put hallucinations to stop-words. Check `misheard text` in `talk-llama.cpp`
- don't put cyrillic (Russian) letters for characters or paths in .bat files, they may not work nice because of weird encoding. Use cmd instead if you need to use cyrillic letters.