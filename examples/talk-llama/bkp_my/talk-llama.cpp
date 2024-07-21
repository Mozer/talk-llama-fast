// talk-llama-fast
// based on https://github.com/ggerganov/whisper.cpp
// XTTS + wav2lip support by Mozer

#include "common-sdl.h"
#include "common.h"
#include "console.h"
#include "console.cpp"
#include "whisper.h"
#include "llama.h"
#include "json.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <sstream>

#include <iostream>

#include <algorithm> 
#include <cctype>
#include <locale>
#include <codecvt>
//#include <mutex>
#include <queue>

// for xtts_play_allowed.txt
#include <Windows.h> // for win. 
// for linux - google yourself




//#include <windows.h>
//#include <sys/types.h>
//#pragma comment(lib,"ws2_32.lib")



#include <clocale>
#include <curl/curl.h>
#include <unordered_set>
#include <ctype.h>
#include <map>
#include <iterator>
#include <ctime>


// global
std::string g_hotkey_pressed = "";

std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    auto * model = llama_get_model(ctx);

    // upper limit for the number of tokens
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos, false);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos, false);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), 0, false);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), 0, false);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

const float* parse_float_list(const std::string& s) {    
	std::vector<float> temp;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            temp.push_back(std::stof(item));
        }
    }
    
    int size = temp.size();
    if (size == 0) {
        return nullptr;  // Возвращаем nullptr для пустого массива
    }
    
    float* result = new float[size];
    std::memcpy(result, temp.data(), size * sizeof(float));
    
    return result;
}

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t voice_ms   = 10000;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t n_gpu_layers = 999;

    float vad_thold  = 0.6f;
    float vad_start_thold  = 0.000270f; // 0 to turn off, you can see your current energy_last (loudness level) when running with --print-energy param
    float vad_last_ms  = 1250;
    float freq_thold = 100.0f;

    bool speed_up       = false;
    bool translate      = false;
    bool print_special  = false;
    bool print_energy   = false;
    bool debug          = false;
    bool no_timestamps  = true;
    bool verbose_prompt = false;
    bool verbose        = false;
    bool use_gpu        = true;
    bool allow_newline  = false;
    bool multi_chars    = false;
    bool xtts_intro     = false;
    bool seqrep         = false;
    bool push_to_talk   = false;
    int split_after     = 0;
    int sleep_before_xtts = 0; // in ms
    int main_gpu = 0; 

    std::string person      = "Georgi";
    std::string bot_name    = "LLaMA";
    std::string xtts_voice  = "emma_1";
    std::string wake_cmd    = "";
    std::string heard_ok    = "";
    std::string language    = "en";
    std::string model_wsp   = "models/ggml-base.en.bin";
    std::string model_llama = "models/ggml-llama-7B.bin";
    std::string speak       = "./examples/talk-llama/speak";
    std::string xtts_control_path = "c:\\DATA\\LLM\\xtts\\xtts_play_allowed.txt";
    std::string xtts_url = "http://localhost:8020/";
    std::string google_url = "http://localhost:8003/";
    std::string prompt      = "";
	std::string instruct_preset = "";
	std::string split_mode = "none";
	const float * tensor_split = nullptr;
	std::map<std::string, std::string> instruct_preset_data = {
		{"system_prompt_prefix", ""},
		{"system_prompt_suffix", ""},
		{"user_message_prefix", ""},
		{"user_message_suffix", ""},
		{"bot_message_prefix", ""},
		{"bot_message_suffix", ""},
		{"stop_sequence", ""}
	};
    std::string fname_out;
    std::string path_session = "";       // path to file for saving/loading model eval state
    std::string stop_words = "";
    int32_t ctx_size = 2048;      
    int32_t batch_size = 64;      
    int32_t n_predict = 64;      
    int32_t min_tokens = 0;      
    float temp = 0.9;      
    float top_k = 40;      
    float top_p = 1.0f;      
    float min_p = 0.0f;      
    float repeat_penalty = 1.10;   
    int repeat_last_n = 256;
};

void whisper_print_usage(int argc, const char ** argv, const whisper_params & params);

bool whisper_params_parse(int argc, const char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"   || arg == "--threads")        { params.n_threads      = std::stoi(argv[++i]); }
        else if (arg == "-vms" || arg == "--voice-ms")       { params.voice_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"   || arg == "--capture")        { params.capture_id     = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")     { params.max_tokens     = std::stoi(argv[++i]); }
        else if (arg == "-ac"  || arg == "--audio-ctx")      { params.audio_ctx      = std::stoi(argv[++i]); }
        else if (arg == "-ngl" || arg == "--n-gpu-layers")   { params.n_gpu_layers   = std::stoi(argv[++i]); }
        else if (arg == "-vth" || arg == "--vad-thold")      { params.vad_thold      = std::stof(argv[++i]); }
        else if (arg == "-vths"|| arg == "--vad-start-thold"){ params.vad_start_thold= std::stof(argv[++i]); }
		else if (arg == "-vlm" || arg == "--vad-last-ms")    { params.vad_last_ms    = std::stoi(argv[++i]); }
        else if (arg == "-fth" || arg == "--freq-thold")     { params.freq_thold     = std::stof(argv[++i]); }
        else if (arg == "-su"  || arg == "--speed-up")       { params.speed_up       = true; }
        else if (arg == "-tr"  || arg == "--translate")      { params.translate      = true; }
        else if (arg == "-ps"  || arg == "--print-special")  { params.print_special  = true; }
        else if (arg == "-pe"  || arg == "--print-energy")   { params.print_energy   = true; }
        else if (arg == "--debug")                           { params.debug          = true; }
        else if (arg == "-vp"  || arg == "--verbose-prompt") { params.verbose_prompt = true; }
        else if (arg == "--verbose")                         { params.verbose = true; }
        else if (arg == "-ng"  || arg == "--no-gpu")         { params.use_gpu        = false; }
        else if (arg == "-p"   || arg == "--person")         { params.person         = argv[++i]; }
        else if (arg == "-bn"   || arg == "--bot-name")      { params.bot_name       = argv[++i]; }
        else if (arg == "--session")                         { params.path_session   = argv[++i]; }
        else if (arg == "-w"   || arg == "--wake-command")   { params.wake_cmd       = argv[++i]; }
        else if (arg == "-ho"  || arg == "--heard-ok")       { params.heard_ok       = argv[++i]; }
        else if (arg == "-l"   || arg == "--language")       { params.language       = argv[++i]; }
        else if (arg == "-mw"  || arg == "--model-whisper")  { params.model_wsp      = argv[++i]; }
        else if (arg == "-ml"  || arg == "--model-llama")    { params.model_llama    = argv[++i]; }
        else if (arg == "-s"   || arg == "--speak")          { params.speak          = argv[++i]; }
        else if (arg == "--ctx_size")                        { params.ctx_size       = std::stoi(argv[++i]); }
        else if (arg == "-b"   || arg == "--batch-size")     { params.batch_size     = std::stoi(argv[++i]); }
        else if (arg == "-n"   || arg == "--n_predict")      { params.n_predict      = std::stoi(argv[++i]); }
        else if (arg == "--temp")     						 { params.temp           = std::stof(argv[++i]); }
        else if (arg == "--top_k")     						 { params.top_k          = std::stof(argv[++i]); }
        else if (arg == "--top_p")     						 { params.top_p          = std::stof(argv[++i]); }
        else if (arg == "--min_p")     						 { params.min_p          = std::stof(argv[++i]); }
        else if (arg == "--repeat_penalty")     			 { params.repeat_penalty = std::stof(argv[++i]); }
        else if (arg == "--repeat_last_n")     			     { params.repeat_last_n  = std::stoi(argv[++i]); }
        else if (arg == "--main-gpu")     			         { params.main_gpu       = std::stoi(argv[++i]); }
        else if (arg == "--split-mode")                      { params.split_mode     = argv[++i]; }
        else if (arg == "--tensor-split")                    { params.tensor_split   = parse_float_list(argv[++i]); }
        else if (arg == "--xtts-voice")                      { params.xtts_voice     = argv[++i]; }
        else if (arg == "--xtts-url")                        { params.xtts_url       = argv[++i]; }
        else if (arg == "--google-url")                      { params.google_url     = argv[++i]; }
        else if (arg == "--xtts-control-path")               { params.xtts_control_path = argv[++i]; }
        else if (arg == "--allow-newline")                   { params.allow_newline  = true; }
        else if (arg == "--multi-chars")                     { params.multi_chars    = true; }
        else if (arg == "--xtts-intro")                      { params.xtts_intro     = true; }
        else if (arg == "--sleep-before-xtts")               { params.sleep_before_xtts = std::stoi(argv[++i]); }
        else if (arg == "--seqrep")                          { params.seqrep         = true; }
        else if (arg == "--push-to-talk")                    { params.push_to_talk   = true; }
        else if (arg == "--split-after")                     { params.split_after    = std::stoi(argv[++i]); }
        else if (arg == "--min-tokens")                      { params.min_tokens     = std::stoi(argv[++i]); }
		else if (arg == "--stop-words")                      { params.stop_words     = argv[++i]; }
		else if (arg == "--instruct-preset")                 { params.instruct_preset= argv[++i]; }
        else if (arg == "--prompt-file")                     {
            std::ifstream file(argv[++i]);
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        }		
        else if (arg == "-f"   || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-ng"  || arg == "--no-gpu")        { params.use_gpu       = false; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, const char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help           [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N      [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -vms N,   --voice-ms N     [%-7d] voice duration in milliseconds\n",              params.voice_ms);
    fprintf(stderr, "  -c ID,    --capture ID     [%-7d] capture device ID\n",                           params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N   [%-7d] maximum number of tokens per audio chunk\n",    params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N    [%-7d] audio context size (0 - all)\n",                params.audio_ctx);
    fprintf(stderr, "  -ngl N,   --n-gpu-layers N [%-7d] number of layers to store in VRAM\n",           params.n_gpu_layers);
    fprintf(stderr, "  -vth N,   --vad-thold N    [%-7.2f] voice avg activity detection threshold\n",    params.vad_thold);
	fprintf(stderr, "  -vths N,  --vad-start-thold N [%-7.6f] vad min level to stop tts, 0: off, 0.000270: default\n",params.vad_start_thold);
    fprintf(stderr, "  -vlm N,   --vad-last-ms N  [%-7d] vad min silence after speech, ms\n",       	 params.vad_last_ms);
    fprintf(stderr, "  -fth N,   --freq-thold N   [%-7.2f] high-pass frequency cutoff\n",                params.freq_thold);
    fprintf(stderr, "  -su,      --speed-up       [%-7s] speed up audio by x2 (not working)\n",          params.speed_up ? "true" : "false");
    fprintf(stderr, "  -tr,      --translate      [%-7s] translate from source language to english\n",   params.translate ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special  [%-7s] print special tokens\n",                        params.print_special ? "true" : "false");
    fprintf(stderr, "  -pe,      --print-energy   [%-7s] print sound energy (for debugging)\n",          params.print_energy ? "true" : "false");
    fprintf(stderr, "  --debug                    [%-7s] print debug info\n",                            params.debug ? "true" : "false");
    fprintf(stderr, "  -vp,      --verbose-prompt [%-7s] print prompt at start\n",                       params.verbose_prompt ? "true" : "false");
    fprintf(stderr, "  --verbose                  [%-7s] print speed\n",                                 params.verbose ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu         [%-7s] disable GPU\n",                                 params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -p NAME,  --person NAME    [%-7s] person name (for prompt selection)\n",          params.person.c_str());
    fprintf(stderr, "  -bn NAME, --bot-name NAME  [%-7s] bot name (to display)\n",                       params.bot_name.c_str());
    fprintf(stderr, "  -w TEXT,  --wake-command T [%-7s] wake-up command to listen for\n",               params.wake_cmd.c_str());
    fprintf(stderr, "  -ho TEXT, --heard-ok TEXT  [%-7s] said by TTS before generating reply\n",         params.heard_ok.c_str());
    fprintf(stderr, "  -l LANG,  --language LANG  [%-7s] spoken language\n",                             params.language.c_str());
    fprintf(stderr, "  -mw FILE, --model-whisper  [%-7s] whisper model file\n",                          params.model_wsp.c_str());
    fprintf(stderr, "  -ml FILE, --model-llama    [%-7s] llama model file\n",                            params.model_llama.c_str());
    fprintf(stderr, "  -s FILE,  --speak TEXT     [%-7s] command for TTS\n",                             params.speak.c_str());
    fprintf(stderr, "  --prompt-file FNAME        [%-7s] file with custom prompt to start dialog\n",     "");
    fprintf(stderr, "  --instruct-preset TEXT     [%-7s] instruct preset to use without .json \n",     	 "");
    fprintf(stderr, "  --session FNAME                   file to cache model state in (may be large!) (default: none)\n");
    fprintf(stderr, "  -f FNAME, --file FNAME     [%-7s] text output file name\n",                       params.fname_out.c_str());
    fprintf(stderr, "   --ctx_size N              [%-7d] Size of the prompt context\n",                  params.ctx_size);
    fprintf(stderr, "  -b N,     --batch-size N   [%-7d] Size of input batch size\n",                    params.batch_size);
    fprintf(stderr, "  -n N,     --n_predict N    [%-7d] Max number of tokens to predict\n",             params.n_predict);
    fprintf(stderr, "  --temp N                   [%-7.2f] Temperature \n",                              params.temp);
    fprintf(stderr, "  --top_k N                  [%-7.2f] top_k \n",                                    params.top_k);
    fprintf(stderr, "  --top_p N                  [%-7.2f] top_p \n",                                    params.top_p);
    fprintf(stderr, "  --min_p N                  [%-7.2f] min_p \n",                                    params.min_p);
    fprintf(stderr, "  --repeat_penalty N         [%-7.2f] repeat_penalty \n",                           params.repeat_penalty);
    fprintf(stderr, "  --repeat_last_n N          [%-7d] repeat_last_n \n",                              params.repeat_last_n);
    fprintf(stderr, "  --main-gpu N               [%-7d] main GPU id, starting from 0 \n",               params.main_gpu);
    fprintf(stderr, "  --split-mode NAME          [%-7s] GPU split mode: 'none' or 'layer'\n",           params.split_mode.c_str());
    fprintf(stderr, "  --tensor-split NAME        [%-7s] Tensor split, list of floats: 0.5,0.5\n",	     params.tensor_split);
    fprintf(stderr, "  --xtts-voice NAME          [%-7s] xtts voice without .wav\n",                     params.xtts_voice.c_str());
    fprintf(stderr, "  --xtts-url TEXT            [%-7s] xtts/silero server URL, with trailing slash\n", params.xtts_url.c_str());
    fprintf(stderr, "  --xtts-control-path FNAME  [%-7s] not used anymore\n",                            params.xtts_control_path.c_str());
	fprintf(stderr, "  --xtts-intro               [%-7s] xtts instant short random intro like Hmmm.\n",  params.xtts_intro ? "true" : "false");
    fprintf(stderr, "  --sleep-before-xtts        [%-7d] sleep llama inference before xtts, ms.\n",      params.sleep_before_xtts);
    fprintf(stderr, "  --google-url TEXT          [%-7s] langchain google-serper server URL, with /\n",  params.google_url.c_str());
    fprintf(stderr, "  --allow-newline            [%-7s] allow new line in llama output\n",              params.allow_newline ? "true" : "false");
    fprintf(stderr, "  --multi-chars              [%-7s] xtts will use same wav name as in llama output\n", params.multi_chars ? "true" : "false");
    fprintf(stderr, "  --push-to-talk             [%-7s] hold Alt to speak\n",							 params.push_to_talk ? "true" : "false");
    fprintf(stderr, "  --seqrep                   [%-7s] sequence repetition penalty, search last 20 in 300\n",params.seqrep ? "true" : "false");
    fprintf(stderr, "  --split-after N            [%-7d] split after first n tokens for tts\n",          params.split_after);
    fprintf(stderr, "  --min-tokens N             [%-7d] min new tokens to output\n",                    params.min_tokens);
	fprintf(stderr, "  --stop-words TEXT          [%-7s] llama stop w: separated by ; \n",               params.stop_words.c_str());
    fprintf(stderr, "\n");
}

// returns seconds since epoch with milliseconds. e.g. 15244.575 (15244 s and 575 ms)
float get_current_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();

    // Convert to milliseconds since the Unix epoch
    auto duration = now.time_since_epoch();
    float millis = (float)std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000;
	
	return millis;
}

std::string transcribe(
        whisper_context * ctx,
        const whisper_params & params,
        const std::vector<float> & pcmf32,
        const std::string prompt_text,
        float & prob,
        int64_t & t_ms) {
    const auto t_start = std::chrono::high_resolution_clock::now();

	//printf("%.3f in transcribe\n", get_current_time_ms());
    prob = 0.0f;
    t_ms = 0;

    std::vector<whisper_token> prompt_tokens;

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    prompt_tokens.resize(1024);
    prompt_tokens.resize(whisper_tokenize(ctx, prompt_text.c_str(), prompt_tokens.data(), prompt_tokens.size()));

    wparams.print_progress   = false;
    wparams.print_special    = params.print_special;
    wparams.print_realtime   = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate        = params.translate;
    wparams.no_context       = true;
    wparams.single_segment   = true;
    wparams.max_tokens       = params.max_tokens;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;

    wparams.prompt_tokens    = prompt_tokens.empty() ? nullptr : prompt_tokens.data();
    wparams.prompt_n_tokens  = prompt_tokens.empty() ? 0       : prompt_tokens.size();

    wparams.audio_ctx        = params.audio_ctx;
    //wparams.speed_up         = params.speed_up;
    wparams.no_timestamps    = true;

	if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        return "";
    }
	
    int prob_n = 0;
    std::string result;

    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);

        result += text;

        const int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; ++j) {
            const auto token = whisper_full_get_token_data(ctx, i, j);

            prob += token.p;
            ++prob_n;
        }
    }
	//printf("%.3f after n_segments\n", get_current_time_ms());

    if (prob_n > 0) {
        prob /= prob_n;
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
	
	// print hex
	//printf("whisper result:\n");
	//for (char ch : result) printf("%X ", ch);
	//printf("\n\n");
				
    return result;
}

std::vector<std::string> get_words(const std::string &txt) {
    std::vector<std::string> words;

    std::istringstream iss(txt);
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }

    return words;
}


std::string getTempDir() {
    TCHAR path_buf[MAX_PATH];
    DWORD ret_val = GetTempPath(MAX_PATH, path_buf);
    if ( ret_val > MAX_PATH || (ret_val == 0) )
    {
        printf("GetTempPath failed");
		return "";
    } else {
        return path_buf;
    }
}

// OLD
// writes to file 0 or 1
// @path: not used anymore, we store in temp
// @xtts_play_allowed: 0=dont play xtts, 1=xtts can play
void allow_xtts_file(std::string path, int xtts_play_allowed) {	
    try{            
		std::string temp_path = getTempDir();
		path = temp_path+"xtts_play_allowed.txt";
        const std::string fileName{path};
        std::ifstream readStream{fileName};
		std::string singleLine;
		bool doesExistAndIsReadable{readStream.good()};
		
		if(!doesExistAndIsReadable){
			printf("Notice: %s file not found. Let's create it.", path.c_str());
			// Create the file with the given value
            std::ofstream writeStream(fileName);
            writeStream << xtts_play_allowed;
            writeStream.flush();
            writeStream.close();            
		}
		else {
			// Read the existing value from the file
			std::getline(readStream, singleLine);
			readStream.close();
			int stored_value = stoi(singleLine);
			if (stored_value != xtts_play_allowed)
			{
				std::ofstream writeStream{fileName};
				writeStream << xtts_play_allowed;
				writeStream.flush();
				writeStream.close();
			}
		}
    } catch(...) {                          // exception handler
    }
}

// trim from start (in place)
inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
inline void trim(std::string &s) {
    rtrim(s);
    ltrim(s);
}

bool IsPunctuationMark(char c) {
    switch (static_cast<unsigned char>(c)) {
        case ',':
            [[fallthrough]];
        case '.':
            [[fallthrough]];
        case '?':
            return true;
		case ':':
            return true;
		case '!':
            return true;	
        default:
            return false;
    }
}

std::string StripPunctuationMarks(const std::string& text) {
    std::string cleanText;
    for (const auto& c : text) {
        if (!IsPunctuationMark(c)) {
           cleanText += c;
        }
    }
    return cleanText;
}

std::string LowerCase(const std::string& text) {
    std::string lowerCasedText;
    for (const auto& c : text) {
        lowerCasedText += std::tolower(c, std::locale());
    }
    return lowerCasedText;
}

// get part of the string that is after the @command (please google weather in london -> weather in london)
std::string ParseCommandAndGetKeyword(std::string textHeardTrimmed, const std::string& command="google") {
   
    textHeardTrimmed = StripPunctuationMarks(textHeardTrimmed); 
    //std::string sanitizedInput = LowerCase(textHeardTrimmed); // LowerCase not working with utf8 non-latin chars
    std::string sanitizedInput = textHeardTrimmed; 
    std::size_t pos = 0;
    bool startsWithPrefix = false;
	static const std::unordered_set<std::string> please_needles{"can you hear me", "Can you hear me", "Are you here", "are you here", "Do you hear me", "do you hear me", "Пожалуйста", "пожалуйста", "Позови", "позови", "ты тут", "Ты тут", "ты здесь", "Ты здесь", "Ты слышишь меня", "ты слышишь меня", "ты меня слышишь", "Ты меня слышишь", "Hey", "hey", "please", "Please", "can you", "Can you", "let's", "Let's", "What do you think", "Что ты думаешь", "что ты думаешь", "Что ты об этом думаешь", "что ты об этом думаешь"};
	std::string result_param = "";
	
	// remove please_needles
	for(const auto& prefix : please_needles) {
		sanitizedInput = ::replace(sanitizedInput, prefix, "");
	}
	trim(sanitizedInput);
	
	//fprintf(stdout, "sanitizedInput: %s\n", sanitizedInput.c_str());
	
	if (command=="google")
	{	
		static const std::unordered_set<std::string> prefixNeedles{"Погугли", "погугли", "угли", "углe", "По гугле", "По угли"};
		// find start prefixNeedles
		for(const auto& prefix : prefixNeedles) {
			if(sanitizedInput.compare(0,prefix.length(),prefix)==0) {
				pos = prefix.length()+1;
				startsWithPrefix = true;
				break;
			}
		}	
	}	
	
	// if not starts with prefixNeedles, find exact command
    if(!startsWithPrefix || pos==std::string::npos) {
		//fprintf(stdout, "goin to search %s in %s\n", command.c_str(), sanitizedInput.c_str());
		pos = sanitizedInput.find(command); // call
		if (pos != std::string::npos) pos = pos + command.size()+1;
		else 
		{
			pos = sanitizedInput.find("Call");
			if (pos != std::string::npos) pos = pos + command.size()+1;
			pos = 0; // command not found
		}
    }
	
	if (command=="call")
	{	
		// russian name fixes: Позови Алису -> Алиса
		int last_n = sanitizedInput.size() - 1;
		if (sanitizedInput[last_n-1] == static_cast<char>(0xD1) && sanitizedInput[last_n] ==  static_cast<char>(0x83)) { sanitizedInput[last_n-1] = 0xD0; sanitizedInput[last_n] = 0xB0; } // у -> а
		else if (sanitizedInput[last_n-1] == static_cast<char>(0xD1) && sanitizedInput[last_n] ==  static_cast<char>(0x8E)) { sanitizedInput[last_n-1] = 0xD0; sanitizedInput[last_n] = 0x8F; } // ю -> я
		//else if (sanitizedInput[last_n-1] == static_cast<char>(0xD0) && sanitizedInput[last_n] ==  static_cast<char>(0xB0)) { sanitizedInput = sanitizedInput.substr(0, sanitizedInput.size()-2); } // Олега -> Олег
		textHeardTrimmed = sanitizedInput;
	}	
	
	result_param = textHeardTrimmed.substr(pos);	
	
    return result_param;	
}

static size_t WriteCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
	((std::string*)userdata)->append((const char*)ptr, size * nmemb);
	return size * nmemb;
}

std::string RemoveTrailingCharacters(const std::string &inputString, const char targetCharacter) {
    auto lastNonTargetPosition = std::find_if(inputString.rbegin(), inputString.rend(), [targetCharacter](auto ch) {
        return ch != targetCharacter;
    });
    return std::string(inputString.begin(), lastNonTargetPosition.base());
}

std::string RemoveTrailingCharactersUtf8(const std::string& inputString, const std::u32string& targetCharacter) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    std::u32string u32_input = converter.from_bytes(inputString);

    auto lastNonTargetPosition = std::find_if(u32_input.rbegin(), u32_input.rend(), [&targetCharacter](char32_t ch) {
        return targetCharacter.find(ch) == std::u32string::npos;
    });

    std::string result = converter.to_bytes(std::u32string(u32_input.begin(), lastNonTargetPosition.base()));
    return result;
}

std::string UrlEncode(const std::string& str) {
    CURL* curl = curl_easy_init();
    if (curl) {
        char* encodedUrl = curl_easy_escape(curl, str.c_str(), str.length());
        std::string escapedUrl(encodedUrl);
        curl_free(encodedUrl);
        curl_easy_cleanup(curl);
        return escapedUrl;
    }
    return {};
}

// blocking
std::string send_curl_json(const std::string &url, const std::map<std::string, std::string>& params) {
    CURL *curl;
    CURLcode res;
    std::string readBuffer;
	
    /* Initialize curl */
    curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize curl");
    }
    
    try {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());        
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);
        
        /* Convert map to query string */
        std::ostringstream oss;
        bool firstParam = true;
		oss << "{";
        for (auto param : params) {
          if (!firstParam) oss<< ',';
          oss << "\"" << param.first << "\":\"" << param.second << "\"";
          firstParam=false;
        };
		oss << "}";
        fprintf(stdout, "send_curl_json: %s\n",oss.str().c_str());
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_slist_append(nullptr, "Content-Type:application/json"));
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, oss.str().c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);		
        res = curl_easy_perform(curl);
             
        if (res != CURLE_OK) {
            throw std::runtime_error(std::string("cURL error: ") + curl_easy_strerror(res));
        } else {
            //std::cout << "Request successful!" << std::endl;
        }
    }
	catch(...) {                          // exception handler
    }
	curl_easy_cleanup(curl);
	
	return readBuffer;
}


// simple curl
std::string send_curl(std::string url)
{
  CURL *curl;
  CURLcode res;
  std::string readBuffer;

  curl = curl_easy_init();
  if(curl) {
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
  }
  
  return readBuffer;
}

int utf8_length(const std::string& str)
{
    int c,i,ix,q;
    for (q=0, i=0, ix=str.length(); i < ix; i++, q++)
    {
        c = (unsigned char) str[i];
        if      (c>=0   && c<=127) i+=0;
        else if ((c & 0xE0) == 0xC0) i+=1;
        else if ((c & 0xF0) == 0xE0) i+=2;
        else if ((c & 0xF8) == 0xF0) i+=3;
        //else if (($c & 0xFC) == 0xF8) i+=4; // 111110bb //byte 5, unnecessary in 4 byte UTF-8
        //else if (($c & 0xFE) == 0xFC) i+=5; // 1111110b //byte 6, unnecessary in 4 byte UTF-8
        else return 0;//invalid utf8
    }
    return q;
}

std::string utf8_substr(const std::string& str, unsigned int start, unsigned int leng)
{
    if (leng==0) { return ""; }
    unsigned int c, i, ix, q, min=std::string::npos, max=std::string::npos;
    for (q=0, i=0, ix=str.length(); i < ix; i++, q++)
    {
        if (q==start){ min=i; }
        if (q<=start+leng || leng==std::string::npos){ max=i; }

        c = (unsigned char) str[i];
        if      (
                 //c>=0   &&
                 c<=127) i+=0;
        else if ((c & 0xE0) == 0xC0) i+=1;
        else if ((c & 0xF0) == 0xE0) i+=2;
        else if ((c & 0xF8) == 0xF0) i+=3;
        //else if (($c & 0xFC) == 0xF8) i+=4; // 111110bb //byte 5, unnecessary in 4 byte UTF-8
        //else if (($c & 0xFE) == 0xFC) i+=5; // 1111110b //byte 6, unnecessary in 4 byte UTF-8
        else return "";//invalid utf8
    }
    if (q<=start+leng || leng==std::string::npos){ max=i; }
    if (min==std::string::npos || max==std::string::npos) { return ""; }
    return str.substr(min,max);
}

std::string translit_en_ru(IN const std::string &str) {
	std::string strRes;   
	std::string szEngABC = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
	std::string szRusABC = "абцдефгхийклмнопкрстуввхйзАБЦДЕФГХИЙКЛМНОПКРСТУВВХЙЗ";
	
	char szRusABC_arr[104];
	int lastIndex = 0;
	for (char ch : szRusABC) 
	{		
		szRusABC_arr[lastIndex] = ch;
		lastIndex++;
	}
	
	for (int i=0; i<str.size(); i++) 
	{
	  size_t pCh = szEngABC.find_first_of(str[i]);
	  if (pCh != std::string::npos) {
		 strRes += szRusABC_arr[2*pCh];
		 strRes += szRusABC_arr[2*pCh+1];
		 //fprintf(stdout, "en:%c, ru:%s, pos: %d\n", str[i], strRes, pCh);
	  } else {
		 strRes += str[i];
	  }
	} 
	
	//printf hex
	//fprintf(stdout, "strRes %s:\n", strRes.c_str());
	//for (char ch : strRes) printf("%X ", ch);
	//printf("\n\n"); 
   
	return strRes;
}

// returns name or ""
std::string find_name(const std::string& str) {
	if (str.size()>=4)
	{
		// Search for '\n' character
		size_t pos = str.find("\n");
		if (pos != std::string::npos && pos + 4 <= str.length()) {
			// If found, search for ':' after '\n'
			size_t endPos = str.find(": ", pos);
			if (endPos == std::string::npos || endPos >= str.length()) {
			  return "";
			}
			
			// Get the subtring before ' :'
			std::string substr = str.substr(pos + 1, endPos - pos - 1);
			// Remove leading and trailing spaces
			while (*substr.begin() == ' ') {
				substr.erase(substr.begin());
			}
			while(*substr.rbegin()==' ') {
				substr.pop_back();
			}
			// Return trimmed substring if its length is less than or equal to 20 characters
			if (substr.length() < 2 || substr.length() > 20) {
				return "";
			} else {
				return substr;
			}
		}
	}
    return "";
}

// Function to convert embedding vector to a concatenated string
std::string emb_to_str(llama_context* ctx_llama, const std::vector<llama_token>& embd) {
    std::string ss;
    for (const auto& token : embd) {
        std::string token_str = llama_token_to_piece(ctx_llama, token);
        ss += token_str;
    }
    return ss;
}


// async curl, but it's still blocking for some reason
// use in a new thread for non-blocking
// doesn't wait for responce
void send_tts_async(std::string text, std::string speaker_wav="emma_1", std::string language="en", std::string tts_url="http://localhost:8020/", int reply_part=0)
{
	// remove (text) and <tag> using regex
	if (text[0] == '(' && text[text.size()-1] != ')') text=+")"; // missing )
	else if (text[0] != '(' && text[text.size()-1] == ')') text="("+text; // missing (
	std::regex paren_regex(R"(\([^()]*\))"); // (text)
	std::regex html_regex(R"(<(.*?)>)"); // // <tag>
    text = std::regex_replace(text, paren_regex, ""); 
    text = std::regex_replace(text, html_regex, "");
	
	trim(text);
	text = ::replace(text, "...", ".");
	text = ::replace(text, "..", ".");
	text = ::replace(text, "…", ".");
	text = ::replace(text, "??", "?");
	text = ::replace(text, "!!!", "!");
	text = ::replace(text, "!!", "!");
	text = ::replace(text, "?!", "?");
	text = ::replace(text, "\"", "");
	text = ::replace(text, ")", "");
	text = ::replace(text, "(", "");
	text = ::replace(text, " ?", "?");
	text = ::replace(text, " !", "!");
	text = ::replace(text, " .", ".");
	text = ::replace(text, " ,", ",");
	
	if (text == "!" || text == "?" || text == ".") text = "";
	else if (text == speaker_wav+": ") text = "";
	else if (text == speaker_wav+":") text = "";
	else if (text[text.size()-1] == ':') text[text.size()-1] = ' ';
	else if (text[text.size()-1] == '-') text[text.size()-1] = ' ';
	else if (text[text.size()-1] == '(') text[text.size()-1] = ' ';
	else if (text[text.size()-1] == ',') text[text.size()-1] = ' ';
	else if (text[text.size()-1] == '.') text[text.size()-1] = ' ';
	speaker_wav = ::replace(speaker_wav, ":", "");
	speaker_wav = ::replace(speaker_wav, "\n", "");
	speaker_wav = ::replace(speaker_wav, "\r", "");
	speaker_wav = ::replace(speaker_wav, "\"", "");
	trim(speaker_wav);
	if (speaker_wav.size()<2) speaker_wav = "emma_1";
	trim(text);
	
	if (text[0] == '!' || text[0] == '?' || text[0] == '.' || text[0] == ',' && text.size()>=2) text = utf8_substr(text, 1, utf8_length(text));
	if (text[text.size()-1] == ',' && text.size()>=2) text = utf8_substr(text, 0, utf8_length(text)-1);
	trim(text);
	
	if (text.back() == ':' && text.length() < 15 && text.find(' ') == std::string::npos) text = "";
	if (text.size()>=1 && text != "." && text != "," && text != "!" && text != "\n")
	{
		trim(text);
		text = ::replace(text, "\r", "");
		text = ::replace(text, "\n", " ");
		text = ::replace(text, "\"", "");
		text = ::replace(text, "..", ".");
		tts_url= tts_url + "tts_to_audio/";
		
		CURL *http_handle;
		CURLM *multi_handle;
		int still_running = 1;
		curl_global_init(CURL_GLOBAL_DEFAULT);
		http_handle = curl_easy_init();
		std::string data = "{\"text\":\""+text+"\", \"language\":\""+language+"\", \"speaker_wav\":\""+speaker_wav+"\", \"reply_part\":\""+std::to_string(reply_part)+"\"}";
		//fprintf(stdout, "\n[data (%s)]\n", data.c_str());
		
		curl_easy_setopt(http_handle, CURLOPT_HTTPHEADER, curl_slist_append(nullptr, "Content-Type:application/json"));
		curl_easy_setopt(http_handle, CURLOPT_URL, tts_url.c_str());
		curl_easy_setopt(http_handle, CURLOPT_POSTFIELDS, data.c_str());
		curl_easy_setopt(http_handle, CURLOPT_VERBOSE, 0L);
		
		std::string responseData;
		curl_easy_setopt(http_handle, CURLOPT_WRITEDATA, &responseData);
		curl_easy_setopt(http_handle, CURLOPT_WRITEFUNCTION, WriteCallback);

		multi_handle = curl_multi_init();
		curl_multi_add_handle(multi_handle, http_handle);

		do {
		  CURLMcode mc = curl_multi_perform(multi_handle, &still_running);
	 
		  if(still_running) mc = curl_multi_poll(multi_handle, NULL, 0, 1000, NULL);// wait for activity, timeout or "nothing" 
	 
		  if(mc)
			break;
		} while(still_running);

		curl_easy_cleanup(http_handle);
		curl_multi_cleanup(multi_handle);
		curl_global_cleanup();
	}
}

std::queue<std::string> input_queue; // global
//std::mutex input_mutex;
bool keyboard_input_running = true; // global, not used yet

void input_thread_func() {
    std::string line;
    std::string buffer;
	bool found_another_line = true;
	
    while (keyboard_input_running) {
		do {
			found_another_line = console::readline(line, false);
			buffer += line;
		} while (found_another_line);
		//printf(" '%s' ", buffer.c_str());
		trim(buffer); // keyboard input
		// if you paste multiple passages from clipboard, make sure to hit Enter after pasting
		// otherwise last passage won't be pasted (bug). Alternatively you can manually add \n after last passage in copied text
        
		//std::lock_guard<std::mutex> lock(input_mutex); // was blocking, TODO if really needed
		input_queue.push(buffer);
		buffer = "";
    }
}

// windows only
bool IsConsoleWindowFocused(HWND cur_window_handle) {
    return (cur_window_handle == GetForegroundWindow());
}

// Stop: Ctrl+Space
// Regenerate: Ctrl+Right
// Delete: Ctrl+Delete
// Reset: Ctrl+R
// modifies global var string g_hotkey_pressed 
void keyboard_shortcut_func(HWND cur_window_handle) {
    bool b_ctr_space_processed = false;
    bool b_ctr_right_processed = false;
    bool b_ctr_delete_processed = false;
    bool b_ctr_r_processed = false;
    bool b_ctr_space_prev = false;
    bool b_ctr_right_prev = false;
    bool b_ctr_delete_prev = false;
    bool b_ctr_r_prev = false;
    bool b_ctr_space = false;
    bool b_ctr_right = false;
    bool b_ctr_delete = false;
    bool b_ctr_r = false;
    bool b_alt = false;
    bool isFocused = false;
	g_hotkey_pressed = "";

    while (true) {
        isFocused = IsConsoleWindowFocused(cur_window_handle);
        if (isFocused) {
            b_ctr_space = GetAsyncKeyState(VK_CONTROL) & GetAsyncKeyState(VK_SPACE) & 0x8000;
            b_ctr_right = GetAsyncKeyState(VK_CONTROL) & GetAsyncKeyState(VK_RIGHT) & 0x8000;
            b_ctr_delete = GetAsyncKeyState(VK_CONTROL) & GetAsyncKeyState(VK_DELETE) & 0x8000;
            b_ctr_r = GetAsyncKeyState(VK_CONTROL) & GetAsyncKeyState('R') & 0x8000;
            b_alt = GetAsyncKeyState(VK_MENU) & 0x8000;
			
			if (b_alt)
			{
				g_hotkey_pressed = "Alt";
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				continue; // no further if-else for Alt PTT
			}			
				
            if (b_ctr_space && !b_ctr_space_prev) {
                if (!b_ctr_space_processed) {
                    fflush(stdout);
					printf("\b"); // remove printed symbols
					fflush(stdout);
					g_hotkey_pressed = "Ctrl+Space";
                    b_ctr_space_processed = true;
                }
            }
            else if (!b_ctr_space && b_ctr_space_prev && b_ctr_space_processed) {
                b_ctr_space_processed = false;
            }

            if (b_ctr_right && !b_ctr_right_prev) {
                if (!b_ctr_right_processed) {
					fflush(stdout);
					printf("\b"); // remove printed symbols
					fflush(stdout);					
					g_hotkey_pressed = "Ctrl+Right";
                    b_ctr_right_processed = true;
                }
            }
            else if (!b_ctr_right && b_ctr_right_prev && b_ctr_right_processed) {
                b_ctr_right_processed = false;
            }

            if (b_ctr_delete && !b_ctr_delete_prev) {
                if (!b_ctr_delete_processed) {
					fflush(stdout);
					printf("\b"); // remove printed symbols
					fflush(stdout);	
					g_hotkey_pressed = "Ctrl+Delete";
                    b_ctr_delete_processed = true;
                }
            }
            else if (!b_ctr_delete && b_ctr_delete_prev && b_ctr_delete_processed) {
                b_ctr_delete_processed = false;
            }

            if (b_ctr_r && !b_ctr_r_prev) {
                if (!b_ctr_r_processed) {
					fflush(stdout);
					printf("\b\b"); // remove printed ^R
					fflush(stdout);
					g_hotkey_pressed = "Ctrl+R";
                    b_ctr_r_processed = true;
                }
            }
            else if (!b_ctr_r && b_ctr_r_prev && b_ctr_r_processed) {
                b_ctr_r_processed = false;
            }

            b_ctr_space_prev = b_ctr_space;
            b_ctr_right_prev = b_ctr_right;
            b_ctr_delete_prev = b_ctr_delete;
            b_ctr_r_prev = b_ctr_r;
		}
		
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


const std::string k_prompt_whisper = R"(A conversation with a person called {1}.)";
const std::string k_prompt_whisper_ru = R"({1}, Алиса.)";

const std::string k_prompt_llama = R"(Text transcript of a never ending dialog, where {0} interacts with an AI assistant named {1}.
{1} is helpful, kind, honest, friendly, good at writing and never fails to answer {0}’s requests immediately and with details and precision.
There are no annotations like (30 seconds passed...) or (to himself), just what {0} and {1} say aloud to each other.
The transcript only includes text, it does not include markup like HTML and Markdown.
{1} responds with short and concise answers.

{0}{4} Hello, {1}!
{1}{4} Hello {0}! How may I help you today?
{0}{4} What time is it?
{1}{4} It is {2} o'clock, {5}, year {3}.
{0}{4} What is a cat?
{1}{4} A cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae.
{0}{4} Name a color.
{1}{4} Blue
{0}{4})";









int run(int argc, const char ** argv) {
	
	whisper_params params;
	std::vector<std::thread> threads;
	std::thread t;
	int thread_i = 0;
	int reply_part = 0;
	std::string text_to_speak_arr[150];
	int reply_part_arr[150];
	bool last_output_has_username = false;	
	bool last_output_has_EOT = true;	
	int input_tokens_count = 0;	
	
	HWND cur_window_handle = GetForegroundWindow(); // i hope you run a window that has focus
	
    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }
	
	allow_xtts_file(params.xtts_control_path, 1); // xtts can play	

    // whisper init
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = params.use_gpu;    

    struct whisper_context * ctx_wsp = whisper_init_from_file_with_params(params.model_wsp.c_str(), cparams);

    // llama init

    llama_backend_init();

    auto lmparams = llama_model_default_params();
    if (!params.use_gpu) {
        lmparams.n_gpu_layers = 0;
    } else {
        lmparams.n_gpu_layers = params.n_gpu_layers;
    }
	
	lmparams.main_gpu = params.main_gpu;
	if (params.split_mode == "layer") lmparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
	else lmparams.split_mode = LLAMA_SPLIT_MODE_NONE;
	lmparams.tensor_split = params.tensor_split;
	
    struct llama_model * model_llama = llama_load_model_from_file(params.model_llama.c_str(), lmparams);

    llama_context_params lcparams = llama_context_default_params();

    // tune these to your liking
    lcparams.n_ctx      = params.ctx_size; // 2048 default
    lcparams.seed       = 1;
    lcparams.n_threads  = params.n_threads;
    lcparams.n_batch    = params.batch_size; // 512 was default

    struct llama_context * ctx_llama = llama_new_context_with_model(model_llama, lcparams);

    // print some info about the processing
    {
        fprintf(stderr, "\n");

        if (!whisper_is_multilingual(ctx_wsp)) {
			fprintf(stderr, "WARNING: model is not multilingual");
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing, %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        fprintf(stderr, "\n");
    }

    // init audio

    audio_async audio(15*1000); // length_ms
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    audio.resume();

    bool is_running  = true;
    bool force_speak = false;

    float prob0 = 0.0f;

    const std::string chat_symb = ":";

    std::vector<float> pcmf32_cur;
    std::vector<float> pcmf32_prev;
    std::vector<float> pcmf32_prompt;

	std::string prompt_whisper;
	if (params.language == "ru") std::string prompt_whisper = ::replace(k_prompt_whisper_ru, "{1}", "Анна"); // Алиса is bad
	else std::string prompt_whisper = ::replace(k_prompt_whisper, "{1}", params.bot_name);
	
    // construct the initial prompt for LLaMA inference
    std::string prompt_llama = params.prompt.empty() ? k_prompt_llama : params.prompt;
	
	// instruct mode	
	if (!params.instruct_preset.empty())
	{
		try {
			std::string filename = "instruct_presets/" + params.instruct_preset + ".json";		
			nlohmann::json jsonData;
			std::ifstream jsonFile(filename);

			if (jsonFile.is_open()) {
				jsonFile >> jsonData;
				jsonFile.close();
				params.instruct_preset_data = jsonData;
			} else { // not found
				std::cout << "Warning: preset file '" << filename << "' does not exist. Turning off instruct mode" << std::endl;
				params.instruct_preset = "";				
			}
		}			
		catch (const std::exception &e) {
			std::cerr << "Error parsing JSON: " << e.what() << std::endl;
			return 1;
		}
	}
	else // not passed
	{
		params.instruct_preset = "";		
	}

    // need to have leading ' '
    prompt_llama.insert(0, 1, ' ');

    prompt_llama = ::replace(prompt_llama, "{0}", params.person);
    prompt_llama = ::replace(prompt_llama, "{1}", params.bot_name);

    {
        // get time string
        std::string time_str;
        {
            time_t t = time(0);
            struct tm * now = localtime(&t);
            char buf[128];
            strftime(buf, sizeof(buf), "%H:%M", now);
            time_str = buf;
        }
        prompt_llama = ::replace(prompt_llama, "{2}", time_str);
    }

    {
        // get year string
        std::string year_str;
        std::string ymd;
        {
            time_t t = time(0);
            struct tm * now = localtime(&t);
            char buf[128];
            strftime(buf, sizeof(buf), "%Y", now);
            year_str = buf;
			strftime(buf, sizeof(buf), "%Y-%m-%d", now);
            ymd = buf;
        }
        prompt_llama = ::replace(prompt_llama, "{3}", year_str);
        prompt_llama = ::replace(prompt_llama, "{5}", ymd);
    }

    prompt_llama = ::replace(prompt_llama, "{4}", chat_symb);

    // init session
    std::string path_session = params.path_session;
    std::vector<llama_token> session_tokens;
    auto embd_inp = ::llama_tokenize(ctx_llama, prompt_llama, true);

    if (!path_session.empty()) {
        fprintf(stderr, "%s: attempting to load saved session from %s\n", __func__, path_session.c_str());

        // fopen to check for existing session
        FILE * fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);

            session_tokens.resize(llama_n_ctx(ctx_llama));
            size_t n_token_count_out = 0;
            if (!llama_load_session_file(ctx_llama, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            for (size_t i = 0; i < session_tokens.size(); i++) {
                embd_inp[i] = session_tokens[i];
            }

            fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
        } else {
            fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
        }
    }

    // evaluate the initial prompt

    printf("\n");
    printf("%s : initializing - please wait ...\n", __func__);
	
	float llama_start_time = get_current_time_ms();	
	
	// NEW prompt eval
	int n_past = 0;
	// Calculate the number of chunks needed
	size_t num_chunks = (embd_inp.size() + lcparams.n_batch - 1) / lcparams.n_batch;
	// Iterate through the chunks and evaluate them
	for (size_t i = 0; i < num_chunks; i++) {
		// Calculate the start and end indices for the current chunk
		size_t start_idx = i * lcparams.n_batch;
		size_t end_idx = std::min((i + 1) * lcparams.n_batch, embd_inp.size());
		size_t chunk_size = end_idx - start_idx;
		// Evaluate the current chunk
		llama_eval(ctx_llama, embd_inp.data() + start_idx, chunk_size, n_past);
		n_past += chunk_size;
	}

	float llama_end_time = get_current_time_ms();
	float llama_time_total = 0;
	float llama_time_input = 0;
	float llama_time_output = 0;
	
	llama_time_total = llama_end_time - llama_start_time;
	printf(" \nLlama start prompt: %d/%d tokens in %.3f s at %.0f t/s\n", embd_inp.size(), params.ctx_size, llama_time_total, embd_inp.size()/llama_time_total);


    if (params.verbose_prompt) {
        fprintf(stdout, "\n");
        fprintf(stdout, "%s", prompt_llama.c_str());
        fflush(stdout);
    }

     // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (session_tokens.size()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (n_matching_session_tokens >= embd_inp.size()) {
            fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            fprintf(stderr, "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }
    }

    // HACK - because session saving incurs a non-negligible delay, for now skip re-saving session
    // if we loaded a session with at least 75% similarity. It's currently just used to speed up the
    // initial prompt so it doesn't need to be an exact match.
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < (embd_inp.size() * 3 / 4);
    

    // show wake command if enabled
    const std::string wake_cmd = params.wake_cmd;
    const int wake_cmd_length = get_words(wake_cmd).size();
    const bool use_wake_cmd = wake_cmd_length > 0;

    if (use_wake_cmd) {
        printf("%s : the wake-up command is: '%s%s%s'\n", __func__, "\033[1m", wake_cmd.c_str(), "\033[0m");
    }

    // clear audio buffer
    audio.clear();

    // text inference variables
    const int voice_id = 2;
    const int n_keep   = embd_inp.size();
    const int n_ctx    = llama_n_ctx(ctx_llama);

    n_past = n_keep;
    int n_prev = 64;
    std::vector<int> past_prev_arr{};
    int n_past_prev = 0; // token count that was before the last answer
    int n_session_consumed = !path_session.empty() && session_tokens.size() > 0 ? session_tokens.size() : 0;
    std::vector<llama_token> embd;
	std::string text_heard_prev;
	std::string text_heard_trimmed;
	int new_command_allowed = 1;
	std::string google_resp;
	std::vector<std::string> tts_intros;
	std::string rand_intro_text = "";
	std::string last_output_buffer = "";
	std::string last_output_needle = "";
	if (params.language == "ru")
	{
		tts_intros = {"Хм", "Ну", "Нуу", "О", "А", "А?", "Угу", "Ох", "Ха", "Ах", "Блин", "Короче", "В общем", "Ой", "Слышь", "Ну вообще-то", "Ну а вообще", "Кароче", "Вот", "Знаешь", "Как бы", "Прикинь", "Послушай", "Типа", "Это", "Так вот", "Погоди", params.person};
	}
	else
	{
		tts_intros = {"Hm", "Hmm", "Well", "Well well", "Huh", "Ugh", "Uh", "Um", "Mmm", "Oh", "Ooh", "Haha", "Ha ha", "Ahh", "Whoa", "Really", "I mean", "By the way", "Anyway", "So", "Actually", "Uh-huh", "Seriously", "Whatever", "Ahh", "Like", "But", "You know", "Wait", "Ahem", "Damn", params.person};
	}
	srand(time(NULL)); // Initialize the random number generator
	
	int last_command_time = 0;
	int eot_antiprompt_id_1 = 0;
	int eot_antiprompt_id_2 = 0;
	std::string current_voice = params.xtts_voice;
	
    // reverse prompts for detecting when it's time to stop speaking
    std::vector<std::string> antiprompts = {
        params.person + chat_symb,
        params.person + " "+chat_symb,
    };
	if (!params.allow_newline) antiprompts.push_back("\n");
	if (!params.instruct_preset_data["stop_sequence"].empty())  antiprompts.push_back(params.instruct_preset_data["stop_sequence"]);
	if (!params.instruct_preset_data["bot_message_suffix"].empty())  
	{
		antiprompts.push_back(params.instruct_preset_data["bot_message_suffix"]);
		eot_antiprompt_id_1 = antiprompts.size()-1;
		antiprompts.push_back("</end_of_turn>"); // weird tag that gemma-2-9 sometimes return instead of <end_of_turn>
		eot_antiprompt_id_2 = antiprompts.size()-1;
	}
	
	// additional stop words
	size_t startIndex = 0;
    size_t endIndex = params.stop_words.find(';');
	if (params.stop_words.size())
	{
		if (endIndex == std::string::npos) // single word
		{
			antiprompts.push_back(params.stop_words);
		}
		else
		{
			while (startIndex < params.stop_words.size()) // multiple stop-words
			{
				std::string word = params.stop_words.substr(startIndex, endIndex - startIndex);
				if (word.size())
				{
					word = ::replace(word, "\\r", "\r");
					word = ::replace(word, "\\n", "\n");
					antiprompts.push_back(word);
				}
				startIndex = endIndex + 1;
				endIndex = params.stop_words.find(';', startIndex);
				if (endIndex == std::string::npos) endIndex = params.stop_words.size();
			}	
		}
	}
	printf("Llama stop words: ");
	for (const auto &prompt : antiprompts) printf("'%s', ", prompt.c_str());

	std::thread input_thread(input_thread_func);
	std::thread shortcut_thread([cur_window_handle]() {
        keyboard_shortcut_func(cur_window_handle);
    });
	printf("\nVoice commands: Stop(Ctrl+Space), Regenerate(Ctrl+Right), Delete(Ctrl+Delete), Reset(Ctrl+R)\n");
	if (params.push_to_talk) printf("Type anything or hold 'Alt' to speak:\n");
	else printf("Start speaking or typing:\n");
	
	printf("\n\n");
    printf("%s%s ", params.person.c_str(), chat_symb.c_str());
    fflush(stdout);
	int vad_result_prev = 2; // ended
	float speech_start_ms = 0;
	float speech_end_ms = 0;
	float speech_len = 0;
	int len_in_samples = 0;
	std::string all_heard_pre;
	int llama_interrupted = 0;
	float llama_interrupted_time = 0.0;	
	llama_start_time = 0.0;
	float llama_start_generation_time = 0.0; //  after prompt processing
	llama_end_time = 0.0;
	llama_time_total = 0.0;
	std::string user_typed = "";
	bool user_typed_this = false;
	
    // main loop	
    while (is_running) {
        // handle Ctrl + C
        is_running = sdl_poll_events();

        if (!is_running) {
            break;
        }

        // delay. try lower?
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        int64_t t_ms = 0;
		
		// keyboard input
		user_typed_this = false;
		console::set_display(console::reset);
		//std::lock_guard<std::mutex> lock(input_mutex); // was blocking, TODO if it is really needed
		if (!input_queue.empty()) {
			std::string buffer;
			while (!input_queue.empty()) {
				buffer += input_queue.front() + "\n";
				input_queue.pop();
			}
			user_typed = buffer;
			trim(user_typed);			
			user_typed_this = true;
		}
		
		// hotkeys
		if (g_hotkey_pressed.size())
		{			
			if (g_hotkey_pressed == "Ctrl+Space") {
				user_typed = "Stop";
			} else if (g_hotkey_pressed == "Ctrl+Right") {
				user_typed = "Regenerate";
			} else if (g_hotkey_pressed == "Ctrl+Delete") {
				user_typed = "Delete";
			} else if (g_hotkey_pressed == "Ctrl+R") {
				user_typed = "Reset";
			}
			
			if (g_hotkey_pressed != "Alt") 
			{
				user_typed_this = true;
				g_hotkey_pressed = "";
			}
		}

		// audio
        {
            audio.get(2000, pcmf32_cur); // step_ms, async			
			
			// WHISPER_SAMPLE_RATE 16000
			// vad_last_ms default 1250
			
			int vad_result = ::vad_simple_int(pcmf32_cur, WHISPER_SAMPLE_RATE, params.vad_last_ms, params.vad_thold, params.freq_thold, params.print_energy, params.vad_start_thold);			
			if (vad_result == 1 && params.vad_start_thold) // speech started
			{
				if (vad_result_prev != 1) // real start
				{					
					speech_start_ms = get_current_time_ms(); // float
					vad_result_prev = 1;
					
					// whisper warmup request, not real one
					//audio.get((int)(speech_len*1000), pcmf32_cur);					
					//printf("%.3f after vad-start, before pre transcribe (%d)\n", get_current_time_ms(), pcmf32_cur.size());
					if (!params.push_to_talk || (params.push_to_talk && g_hotkey_pressed == "Alt"))
					{
						all_heard_pre = ::trim(::transcribe(ctx_wsp, params, pcmf32_cur, prompt_whisper, prob0, t_ms));	// warmup - try with small size audio
						g_hotkey_pressed = "";
					}
					//printf("%.3f after pre transcribe (%d)\n", get_current_time_ms(), pcmf32_cur.size());	
				}
				// user has started speaking, xtts cannot play
				if (!params.push_to_talk || (params.push_to_talk && g_hotkey_pressed == "Alt"))
				{
					allow_xtts_file(params.xtts_control_path, 0);
				}
			}		
            if (vad_result >= 2 && vad_result_prev == 1 || force_speak || user_typed.size())  // speech ended or user typed
			{
				speech_end_ms = get_current_time_ms(); // float in seconds.ms
				speech_len = speech_end_ms - speech_start_ms;
				if (speech_len < 0.10) speech_len = 0;
				else if (speech_len > 10.0) speech_len = 0;
				//printf("%.3f found vad length: %.2f\n", get_current_time_ms(), speech_len);
				//len_in_samples = (int)(WHISPER_SAMPLE_RATE * speech_len);
				//if (len_in_samples && len_in_samples < pcmf32_cur.size())
				//{
				//	std::vector<float> temp(pcmf32_cur.end() - len_in_samples, pcmf32_cur.end()); 
				//	pcmf32_cur.assign(temp.begin(), temp.end());
				//	printf("%.3f trimmed pcmf32_cur up to last %d samples\n", get_current_time_ms(), len_in_samples);
				//}
				vad_result_prev = 2;
				speech_start_ms = 0;
				
				if (!speech_len && !user_typed.size()) continue;
				
				speech_len = speech_len + 0.3; // front padding
				if (speech_len < 1.10) speech_len = 1.10; // whisper doesn't like sentences < 1.10s
                //audio.get((int)(speech_len*1000), pcmf32_cur);	// was 10000 ms			
                audio.get(10000, pcmf32_cur);	// was 10000 ms			
                std::string all_heard;
                //printf("%.3f after vad-end (%d)\n", get_current_time_ms(), pcmf32_cur.size());
				if (user_typed.size())
				{
					all_heard = user_typed;
					user_typed = "";
				}
				else if (!force_speak) 
				{
					if (!params.push_to_talk || (params.push_to_talk && g_hotkey_pressed == "Alt"))
					{
						all_heard = ::trim(::transcribe(ctx_wsp, params, pcmf32_cur, prompt_whisper, prob0, t_ms)); // real transcribe
						g_hotkey_pressed = "";
					}
                }
				//printf("%.3f after real whisper\n", get_current_time_ms());

                const auto words = get_words(all_heard);

                std::string wake_cmd_heard;
                std::string text_heard;

                for (int i = 0; i < (int) words.size(); ++i) {
                    if (i < wake_cmd_length) {
                        wake_cmd_heard += words[i] + " ";
                    } else {
                        text_heard += words[i] + " ";
                    }
                }					
				if (params.print_energy) fprintf(stdout, " [text_heard: (%s)]\n", text_heard.c_str());				
				
                // check if audio starts with the wake-up command if enabled
                if (use_wake_cmd) {
                    const float sim = similarity(wake_cmd_heard, wake_cmd);

                    if ((sim < 0.7f) || (text_heard.empty())) {
                        audio.clear();
                        continue;
                    }
                }

                // optionally give audio feedback that the current text is being processed
                if (!params.heard_ok.empty()) {
                    int ret = system((params.speak + " " + std::to_string(voice_id) + " '" + params.heard_ok + "'").c_str());
                    if (ret != 0) {
                        fprintf(stderr, "%s: failed to speak\n", __func__);
                    }
                }
				//printf(" [PRE text_heard: (%s)]\n", text_heard.c_str());
                // remove text between brackets using regex
                {
                    std::regex re("\\[.*?\\]");
                    text_heard = std::regex_replace(text_heard, re, "");
                }
				
                // remove text between brackets using regex
                {
                    std::regex re("\\(.*?\\)");
                    text_heard = std::regex_replace(text_heard, re, "");
                }
                // remove all characters, except for letters, numbers, punctuation and ':', '\'', '-', ' '
                if (params.language == "en" && !user_typed_this) text_heard = std::regex_replace(text_heard, std::regex("[^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-]"), ""); // breaks non latin text, e.g. Russian
                // take first line
                text_heard = text_heard.substr(0, text_heard.find_first_of('\n'));

                // remove leading and trailing whitespace
                text_heard = std::regex_replace(text_heard, std::regex("^\\s+"), "");
                text_heard = std::regex_replace(text_heard, std::regex("\\s+$"), "");

				// misheard text, sometimes whisper is hallucinating
				text_heard = RemoveTrailingCharactersUtf8(text_heard, U"!");
				text_heard = RemoveTrailingCharactersUtf8(text_heard, U",");
				text_heard = RemoveTrailingCharactersUtf8(text_heard, U".");
				text_heard = RemoveTrailingCharactersUtf8(text_heard, U"»");
				text_heard = RemoveTrailingCharactersUtf8(text_heard, U"[");
				text_heard = RemoveTrailingCharactersUtf8(text_heard, U"]");
				if (text_heard[0] == '.') text_heard.erase(0, 1);
				if (text_heard[0] == '!') text_heard.erase(0, 1);
				if (text_heard[0] == '[') text_heard.erase(0, 1);
				trim(text_heard);
				if (text_heard == "!" || text_heard == "." || text_heard == "Sil" || text_heard == "Bye" || text_heard == "Okay" || text_heard == "Okay." || text_heard == "Thank you." || text_heard == "Thank you" || text_heard == "Thanks." || text_heard == "Bye." || text_heard == "Thank you for listening." || text_heard == "К" || text_heard == "Спасибо" || text_heard == "Пока" || text_heard == params.bot_name || text_heard == "*Звук!*" || text_heard == "Р" || text_heard.find("Редактор субтитров")!= std::string::npos || text_heard.find("можешь это сделать")!= std::string::npos || text_heard.find("Как дела?")!= std::string::npos || text_heard.find("Это")!= std::string::npos || text_heard.find("Добро пожаловать")!= std::string::npos || text_heard.find("Спасибо за внимание")!= std::string::npos || text_heard.find("Будьте здоровы")!= std::string::npos || text_heard.find("Продолжение следует")!= std::string::npos || text_heard.find("End of")!= std::string::npos || text_heard.find("The End")!= std::string::npos || text_heard.find("THE END")!= std::string::npos || text_heard.find("The film was made")!= std::string::npos || text_heard.find("Translated by")!= std::string::npos || text_heard.find("Thanks for watching")!= std::string::npos || text_heard.find("The second part of the video")!= std::string::npos || text_heard.find("Thank you for watching")!= std::string::npos || text_heard.find("*click*")!= std::string::npos || text_heard.find("Субтитры")!= std::string::npos || text_heard.find("До свидания")!= std::string::npos || text_heard.find("До новых встреч")!= std::string::npos || text_heard.find("ПЕСНЯ")!= std::string::npos || text_heard.find("Silence")!= std::string::npos || text_heard.find("Поехали")!= std::string::npos || text_heard == "You're" || text_heard == "you're" || text_heard == "You're not" || text_heard == "See?" || text_heard == "you" || text_heard == "You" || text_heard == "Yeah" || text_heard == "Well" || text_heard == "Hey" || text_heard == "Oh" || text_heard == "Right" || text_heard == "Real" || text_heard == "Huh" || text_heard == "I" || text_heard == "I'm" || text_heard.find("*звук!")!= std::string::npos || text_heard == "В" || text_heard == "а" || text_heard == "*") text_heard = "";
				text_heard = std::regex_replace(text_heard, std::regex("\\s+$"), ""); // trailing whitespace

				
				//printf("Number of tokens in embd: %zu\n", embd.size());
				//printf("n_past_prev: %d\n", n_past_prev);
				//printf("text_heard_prev: %s\n", text_heard_prev);				
				
				text_heard_trimmed = text_heard; // no periods or spaces
				trim(text_heard_trimmed);
				if (text_heard_trimmed[0] == '.') text_heard_trimmed.erase(0, 1);
				if (text_heard_trimmed[0] == '!') text_heard_trimmed.erase(0, 1);
				if (text_heard_trimmed[text_heard_trimmed.length() - 1] == '.' || text_heard_trimmed[text_heard_trimmed.length() - 1] == '!') text_heard_trimmed.erase(text_heard_trimmed.length() - 1, 1);
				trim(text_heard_trimmed);
				text_heard_trimmed = LowerCase(text_heard_trimmed); // not working right with utf and russian
				
				//fprintf(stdout, " [text_heard: (%s)]\n", text_heard.c_str());
				//fprintf(stdout, "text_heard_trimmed: %s%s%s", "\033[1m", text_heard_trimmed.c_str(), "\033[0m");
                fflush(stdout);
				std::string user_command;
				
				if (params.vad_start_thold)
				{
					// user has finished speaking, xtts can play
					// xtts slows down whisper -> moved this block after whisper
					allow_xtts_file(params.xtts_control_path, 1);
				}
				
				// TTS rand INTRO sentence for instant reponse
				if (params.xtts_intro)
				{
					if (text_heard_trimmed.size())
					{
						rand_intro_text = tts_intros[rand() % tts_intros.size()];					
						text_to_speak_arr[thread_i] = rand_intro_text;								
						threads.emplace_back([&] // creates a thread, threads are cleaned after user ends speech, after n threads
						{
							if (text_to_speak_arr[thread_i-1].size())
							{
								send_tts_async(text_to_speak_arr[thread_i-1], current_voice, params.language, params.xtts_url);
								text_to_speak_arr[thread_i-1] = "";
							}
						});
						thread_i++;	
					}
				}
				
				if (text_heard_trimmed.find("regenerate") != std::string::npos || text_heard_trimmed.find("Переделай") != std::string::npos || text_heard_trimmed.find("Пожалуйста, переделай") != std::string::npos || text_heard_trimmed.find("егенерируй") != std::string::npos || text_heard_trimmed.find("егенерировать") != std::string::npos) user_command = "regenerate";
				
				else if (text_heard_trimmed.find("google") != std::string::npos || text_heard_trimmed.find("Погугли") != std::string::npos || text_heard_trimmed.find("Пожалуйста, погугли") != std::string::npos || text_heard_trimmed.find("По гугл") != std::string::npos) user_command = "google";
				
				else if (text_heard_trimmed.find("reset") != std::string::npos || text_heard_trimmed.find("delete everything") != std::string::npos || text_heard_trimmed.find("Сброс") != std::string::npos || text_heard_trimmed.find("Сбросить") != std::string::npos || text_heard_trimmed.find("Удали все") != std::string::npos || text_heard_trimmed.find("Удалить все") != std::string::npos) user_command = "reset";
				
				else if (text_heard_trimmed.find("delete") != std::string::npos || text_heard_trimmed.find("please do it") != std::string::npos || text_heard_trimmed.find("Удали") != std::string::npos || text_heard_trimmed.find("Удалить сообщение") != std::string::npos || text_heard_trimmed.find("Удали сообщение") != std::string::npos || text_heard_trimmed.find("Удали два сообщения") != std::string::npos || text_heard_trimmed.find("Удали три сообщения") != std::string::npos) user_command = "delete";
				
				else if (text_heard_trimmed == "step" ||  text_heard_trimmed.find("stop") != std::string::npos || text_heard_trimmed.find("Стоп") != std::string::npos || text_heard_trimmed.find("Остановись") != std::string::npos || text_heard_trimmed.find("тановись") != std::string::npos || text_heard_trimmed.find("Хватит") != std::string::npos || text_heard_trimmed.find("Становись") != std::string::npos) user_command = "stop";
				
				else if (text_heard_trimmed.find("call") == 0 || text_heard_trimmed.find("can you call") != std::string::npos || text_heard_trimmed.find("let's call") != std::string::npos || text_heard_trimmed.find("please call") != std::string::npos || text_heard_trimmed.find("can you hear me") != std::string::npos || text_heard_trimmed.find("do you hear me") != std::string::npos || text_heard_trimmed.find("are you here") != std::string::npos || (text_heard_trimmed.find("what do you think") != std::string::npos && text_heard_trimmed.find("what do you think of") == std::string::npos) || text_heard_trimmed.find("Позови") != std::string::npos || text_heard_trimmed.find("Позови пожалуйста") != std::string::npos || text_heard_trimmed.find("позови") != std::string::npos || text_heard_trimmed.find("ты тут") != std::string::npos || text_heard_trimmed.find("Ты тут") != std::string::npos || text_heard_trimmed.find("ты меня слышишь") != std::string::npos || text_heard_trimmed.find("Ты меня слышишь") != std::string::npos || text_heard_trimmed.find("ты слышишь меня") != std::string::npos || text_heard_trimmed.find("Ты слышишь меня") != std::string::npos || text_heard_trimmed.find("Ты здесь") != std::string::npos || text_heard_trimmed.find("ты здесь") != std::string::npos || (text_heard_trimmed.find("то ты думаешь") != std::string::npos && text_heard != "Что ты думаешь?" && text_heard_trimmed.find("то ты думаешь об") == std::string::npos) || (text_heard_trimmed.find("то ты об этом думаешь") != std::string::npos && text_heard != "Что ты об этом думаешь?") ) user_command = "call";
				
				
				if (user_command.size() && !new_command_allowed && std::time(0)-last_command_time >= 2) 
				{
					new_command_allowed = 1; // timeout before same command (whisper hallucinates sometimes)
					//printf("new_command_allowed: %d\n", new_command_allowed);
				}
				
				// REGEN
				if (user_command == "regenerate" || text_heard_trimmed == "Please regenerate" || text_heard_trimmed == "Regenerate please" || text_heard_trimmed == "Regenerate, please" || text_heard_trimmed == "Try again please" || text_heard_trimmed == "Try again, please" || text_heard_trimmed == "Please try again" || text_heard_trimmed == "Try again") 
				{
					if (new_command_allowed)
					{
						new_command_allowed = 0;
						last_command_time = std::time(0);
						if (!past_prev_arr.empty())
						{
							// regenerate prev llama reply
							n_past_prev = past_prev_arr.back();
							past_prev_arr.pop_back();
							int rollback_num = embd_inp.size()-n_past_prev;
							if (rollback_num)
							{						
								embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
								printf(" [regenerating %d tokens. Context: %d]\n", rollback_num, embd_inp.size());
								n_past -= rollback_num;
								text_heard = text_heard_prev;
								text_heard_trimmed = "";								
								text_to_speak_arr[thread_i] = "Regenerating";								
								threads.emplace_back([&] // creates a thread
								{
									if (text_to_speak_arr[thread_i-1].size())
									{
										send_tts_async(text_to_speak_arr[thread_i-1], current_voice, params.language, params.xtts_url);
										text_to_speak_arr[thread_i-1] = "";
									}
								});
								thread_i++;
							}
						}						
					}
				}
				// DELETE
				else if (user_command == "delete" || text_heard_trimmed == "Please delete" || text_heard_trimmed == "Please delete the last message" || text_heard_trimmed == "Delete please" || text_heard_trimmed == "Delete, please") 
				{
					if (new_command_allowed) // delete prev user question and llama reply, then dont do anything
					{
						if (!past_prev_arr.empty())
						{
							if (text_heard_trimmed == "delete two messages" || text_heard_trimmed == "Удали 2 сообщения" || text_heard_trimmed == "Удали два сообщения" ||  text_heard_trimmed == "Please donate to the messages")
							{
								n_past_prev = past_prev_arr.back();
								past_prev_arr.pop_back();
							}
							else if (text_heard_trimmed == "delete three messages" || text_heard_trimmed == "Удали 3 сообщения" || text_heard_trimmed == "Удали три сообщения")
							{
								n_past_prev = past_prev_arr.back();
								past_prev_arr.pop_back();
								n_past_prev = past_prev_arr.back();
								past_prev_arr.pop_back();
							}
						
							n_past_prev = past_prev_arr.back();
							past_prev_arr.pop_back();
							int rollback_num = embd_inp.size()-n_past_prev;
							if (rollback_num)
							{
								embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
								printf(" deleting %d tokens. Tokens in ctx: %d\n", rollback_num, embd_inp.size());
								n_past -= rollback_num;
								text_heard = "";
								text_heard_trimmed = "";
								last_command_time = std::time(0);
								new_command_allowed = 0;
											
								text_to_speak_arr[thread_i] = "Deleted";								
								threads.emplace_back([&] // creates a thread
								{
									if (text_to_speak_arr[thread_i-1].size())
									{
										send_tts_async(text_to_speak_arr[thread_i-1], current_voice, params.language, params.xtts_url);
										text_to_speak_arr[thread_i-1] = "";
									}
								});
								thread_i++;						
							}
						}
						else 
						{
							printf("Nothing to delete more\n");
							send_tts_async("Nothing to delete more", "ux", params.language);
						}
					}
					audio.clear();
				}
				// RESET
				else if (user_command == "reset") 
				{
					if (new_command_allowed)
					{
						if (!past_prev_arr.empty())
						{
							// delete everything except start prompt
							n_past_prev = past_prev_arr.front();
							past_prev_arr.clear();
							int rollback_num = embd_inp.size()-n_past_prev;
							if (rollback_num)
							{
								printf(" [Resetting context of %d tokens.]\n", embd_inp.size());						
								embd_inp = ::llama_tokenize(ctx_llama, prompt_llama, true);
								n_past = 0;
								// NEW prompt eval
								// Calculate the number of chunks needed
								size_t num_chunks = (embd_inp.size() + lcparams.n_batch - 1) / lcparams.n_batch;
								// Iterate through the chunks and evaluate them
								for (size_t i = 0; i < num_chunks; i++) {
									// Calculate the start and end indices for the current chunk
									size_t start_idx = i * lcparams.n_batch;
									size_t end_idx = std::min((i + 1) * lcparams.n_batch, embd_inp.size());
									size_t chunk_size = end_idx - start_idx;
									// Evaluate the current chunk
									llama_eval(ctx_llama, embd_inp.data() + start_idx, chunk_size, n_past);
									n_past += chunk_size;
								}
								printf(" [Context is now %d/%d tokens]\n", embd_inp.size(), params.ctx_size);		
								
								//embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());						
								//n_past -= rollback_num;
								text_heard = "";
								text_heard_trimmed = "";
								send_tts_async("Reset whole context", params.xtts_voice, params.language, params.xtts_url);
								new_command_allowed = 0;
							}
						}
						else 
						{
							printf(" [Nothing to reset more]\n");							
							send_tts_async("Nothing to reset more", params.xtts_voice, params.language, params.xtts_url);
						}
					}
					audio.clear();
					continue;
				}
				// STOP
				else if (user_command == "stop") 
				{
					printf(" [Stopped!]\n");					
					text_heard = "";
					text_heard_trimmed = "";
					audio.clear();
					allow_xtts_file(params.xtts_control_path, 0);
					user_typed = "";
					user_typed_this = false;
					continue;
				}
				// GOOGLE
				else if (user_command == "google") 
				{
					std::string q = ParseCommandAndGetKeyword(text_heard_trimmed, user_command);
					if (q.size())
					{
						std::string url = params.google_url+"google?q="+UrlEncode(q);
						google_resp = send_curl(url);
						if (google_resp.size())
						{
							fprintf(stdout, "google_resp (%s): %s\n", q.c_str(), google_resp.c_str());
							if (google_resp.length() > 200) 
							{
								size_t space_pos = google_resp.find(' ', 201);   
								if(space_pos != std::string::npos) google_resp = google_resp.substr(0, space_pos);   
							}
							google_resp = "Google: "+google_resp+" .";
							text_heard += "\n"+google_resp;
							
							if (google_resp.size()) 
							{
								threads.emplace_back([&] // creates and starts a thread
								{
									if (google_resp.size()) send_tts_async(google_resp, "Google", params.language, params.xtts_url);
								});
								thread_i++;
							}
						}
						printf("\nError: bad google resp for (%s), check that langchain server is ok\n",q.c_str());
					}
					else fprintf(stdout, "can't get search keyword from text_heard_trimmed: %s\n", text_heard_trimmed.c_str());
				}
				// CALL
				else if (user_command == "call") 
				{
					std::string q = ParseCommandAndGetKeyword(text_heard, user_command);
					if (q.size())
					{				
						//fprintf(stdout, "found bot name %s\n", q);
						params.bot_name = q;
					}
					else fprintf(stdout, "\nError: can't find bot name in text_heard_trimmed: %s\n", text_heard_trimmed.c_str());
				}
				
				int translation_is_going = 0;
				int n_embd_inp_before_trans = 0;
				int tokens_in_reply = 0;
				std::string current_voice_tmp = "";
				reply_part = 0;
				
				// LLAMA
				llama_start_time = get_current_time_ms();
                const std::vector<llama_token> tokens = llama_tokenize(ctx_llama, text_heard.c_str(), false);

                if (text_heard.empty() || tokens.empty() || force_speak) {
                    //fprintf(stdout, "%s: Heard nothing, skipping ...\n", __func__);
                    audio.clear();
					g_hotkey_pressed = "";

                    continue;
                }
				
                force_speak = false;
				trim(text_heard);
				//fprintf(stdout, "text_heard %s:\n", text_heard.c_str());
				//for (char ch : text_heard) printf("%X ", ch);
				//printf("\n\n"); 
				
				text_heard_prev = text_heard;
				n_past_prev = embd_inp.size();
				past_prev_arr.push_back(embd_inp.size());
				std::string translation_full = "";
				std::string bot_name_current = params.bot_name;
				std::string bot_name_current_ru = params.bot_name;
				std::string text_heard_with_instruct = text_heard;
				if (params.translate) bot_name_current_ru = translit_en_ru(params.bot_name);
				int n_comas = 0; // comas counter
				//printf("text_heard_prev: %s\n", text_heard_prev);
                
				
				if (last_output_has_username && !user_typed_this) // last model output has user name
				{
					text_heard.insert(0, 1, ' '); // missing space
					text_heard_with_instruct.insert(0, 1, ' '); // missing space
				}
				else if (!last_output_has_EOT) // no EOT ( <end_of_turn>)
                {
					text_heard.insert(0, "\n"+params.person + chat_symb + " ");
					text_heard_with_instruct.insert(0, params.instruct_preset_data["bot_message_suffix"] +"\n"+ params.instruct_preset_data["user_message_prefix"]+"\n"+params.person + chat_symb + " ");
				}
				else // has EOT or no_instuct
				{
					text_heard.insert(0, "\n"+params.person + chat_symb + " ");
					text_heard_with_instruct.insert(0, "\n"+params.instruct_preset_data["user_message_prefix"]+"\n"+params.person + chat_symb + " ");
				}
                
				text_heard += "\n" + params.bot_name + chat_symb;
				text_heard_with_instruct += params.instruct_preset_data["user_message_suffix"]+"\n" + params.instruct_preset_data["bot_message_prefix"]+ "\n" + params.bot_name + chat_symb;
                
				if (user_typed_this) 
				{
					fprintf(stdout, "%s%s%s", "\033[1m", params.bot_name + chat_symb, "\033[0m");
					g_hotkey_pressed = "";
				}
				else fprintf(stdout, "%s%s%s", "\033[1m", text_heard.c_str(), "\033[0m");
				if (params.instruct_preset.size()) text_heard = text_heard_with_instruct; // don't print instruct, using another string var	
                fflush(stdout);
				int split_after = params.split_after;
				
                embd = ::llama_tokenize(ctx_llama, text_heard, false); // not sure why 2 times llama_tokenize
				input_tokens_count = embd.size();

                // Append the new input tokens to the session_tokens vector
                if (!path_session.empty()) {
                    session_tokens.insert(session_tokens.end(), tokens.begin(), tokens.end());
                }
				
				// removing all threads
				if (threads.size() >=80) // check text_to_speak_arr init size 150
				{
					printf("[!...", threads.size());
					for (auto& t : threads) 
					{
						try 
						{
							if (t.joinable()) t.join();							
							else printf("Notice: thread %d is NOT joinable\n", thread_i );
						}
						catch (const std::exception& ex) {
							std::cerr << "[Exception]: Failed join a thread: " << ex.what() << '\n';
						}
					}
					threads.clear();
					printf("]");
				}
				if (thread_i > 100) thread_i = 0; // rotation
				
				float temp_next = params.temp;

                // text inference
                bool done = false;
                std::string text_to_speak;
				int new_tokens = 0;	   
                while (true) {
                    // predict
					if (new_tokens > params.n_predict) break; // 64 default
					new_tokens++;
                    if (embd.size() > 0) {
                        if (n_past + (int) embd.size() > n_ctx) {
                            n_past = n_keep;

                            // insert n_left/2 tokens at the start of embd from last_n_tokens
                            embd.insert(embd.begin(), embd_inp.begin() + embd_inp.size() - n_prev, embd_inp.end());
                            // stop saving session if we run out of context
                            path_session = "";
                            //printf("\n---\n");
                            //printf("resetting: '");
                            //for (int i = 0; i < (int) embd.size(); i++) {
                            //    printf("%s", llama_token_to_piece(ctx_llama, embd[i]));
                            //}
                            //printf("'\n");
                            //printf("\n---\n");
                        }

                        // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
                        // REVIEW
                        if (n_session_consumed < (int) session_tokens.size()) {
                            size_t i = 0;
                            for ( ; i < embd.size(); i++) {
                                if (embd[i] != session_tokens[n_session_consumed]) {
                                    session_tokens.resize(n_session_consumed);
                                    break;
                                }

                                n_past++;
                                n_session_consumed++;

                                if (n_session_consumed >= (int) session_tokens.size()) {
                                    i++;
                                    break;
                                }
                            }
                            if (i > 0) {
                                embd.erase(embd.begin(), embd.begin() + i);
                            }
                        }

                        if (embd.size() > 0 && !path_session.empty()) {
                            session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                            n_session_consumed = session_tokens.size();
                        }
						
						// OLD
                        //if (llama_eval(ctx_llama, embd.data(), embd.size(), n_past)) {
                        //    fprintf(stderr, "%s : failed to eval\n", __func__);
                        //    return 1;
                        //}
						
						// NEW prompt eval
						size_t total_size = embd.size();
						//size_t batch_size = lcparams.n_batch;
						// Split the input embeddings into smaller chunks
						for (size_t i = 0; i < total_size; i += lcparams.n_batch) {
							size_t chunk_size = (total_size - i < lcparams.n_batch) ? (total_size - i) : lcparams.n_batch;
							llama_eval(ctx_llama, embd.data() + i, chunk_size, n_past);
							n_past += chunk_size;
						}
                    }	
					

                    embd_inp.insert(embd_inp.end(), embd.begin(), embd.end());
                    //n_past += embd.size();

                    embd.clear();
                    if (done) break;
					std::string out_token_str = "";
					char out_token_symbol;
					
					if (!llama_start_generation_time) llama_start_generation_time = get_current_time_ms();
									
                    {
                        // out of user input, sample next token
                        const float top_k          = params.top_k;
                        const float top_p          = params.top_p;
                        const float min_p          = params.min_p;
                        float temp                 = params.temp;                       
                        const float repeat_penalty = params.repeat_penalty;
                        const int repeat_last_n    = params.repeat_last_n; // 256

                        if (!path_session.empty() && need_to_save_session) {
                            need_to_save_session = false;
                            llama_save_session_file(ctx_llama, path_session.c_str(), session_tokens.data(), session_tokens.size());
                        }

                        llama_token id = 0;	
						int person_name_is_found = 0;						
						int bot_name_is_found = 0;						
						
                        {
                            auto logits = llama_get_logits(ctx_llama);
                            auto n_vocab = llama_n_vocab(model_llama);

                            logits[llama_token_eos(model_llama)] = 0;

                            std::vector<llama_token_data> candidates;
                            candidates.reserve(n_vocab);
                            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                            }

                            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                            // apply repeat penalty
                            const float nl_logit = logits[llama_token_nl(model_llama)];

                            llama_sample_repetition_penalties(ctx_llama, &candidates_p,
                                    embd_inp.data() + std::max(0, n_past - repeat_last_n),
                                    repeat_last_n, repeat_penalty, 0.0, 0.0f);

                            logits[llama_token_nl(model_llama)] = nl_logit;
							
							if (temp != temp_next) temp = temp_next; // high temp just for 1 token
	
                            if (temp <= 0) {
                                // Greedy sampling
                                id = llama_sample_token_greedy(ctx_llama, &candidates_p);
                            } else {
                                // Temperature sampling
                                llama_sample_top_k(ctx_llama, &candidates_p, top_k, 1);
                                llama_sample_top_p(ctx_llama, &candidates_p, top_p, 1);
                                llama_sample_min_p(ctx_llama, &candidates_p, min_p, 1);
                                llama_sample_temp (ctx_llama, &candidates_p, temp);
                                id = llama_sample_token(ctx_llama, &candidates_p);
                            }
							//if (!tokens_in_reply) printf("%.3f after 1st token\n", get_current_time_ms());
							if (temp != params.temp) temp = temp_next = params.temp; // back to normal temp
                        }
						
                        if (id != llama_token_eos(model_llama)) {
                            // add it to the context
                            embd.push_back(id);
							out_token_str = llama_token_to_piece(ctx_llama, id);
                            text_to_speak += out_token_str;
                            //printf("[%s]", out_token_str.c_str());
                            printf("%s", out_token_str.c_str());
							tokens_in_reply++;
							
							
							// detect sequence repetition (llama stuck in a loop) https://github.com/ggerganov/llama.cpp/pull/2593
							if (params.seqrep)
							{
								// fill new needle
								if (utf8_length(last_output_needle) > 25) last_output_needle = utf8_substr(last_output_needle, 5, utf8_length(last_output_needle)-5);
								last_output_needle += out_token_str;								
								out_token_symbol = out_token_str[out_token_str.size()-1] ;
								if (out_token_symbol == ' ' || out_token_symbol == '.' || out_token_symbol == ',' || out_token_symbol == '!' || out_token_symbol == '?')
								{
									// search old needle
									if (utf8_length(last_output_buffer) > 300 && utf8_length(last_output_needle) >= 20 && last_output_buffer.find(last_output_needle) != std::string::npos)
									{
										printf(" [LOOP: %s]\n", last_output_needle.c_str(), utf8_length(last_output_needle));
										
										int symbols_to_delete = static_cast<int>(utf8_length(last_output_needle) * 1); //  all the needle, todo: check params.person and bot name
										const std::vector<llama_token> tokens_to_del = llama_tokenize(ctx_llama, last_output_needle.c_str(), false);
										int rollback_num = tokens_to_del.size();
										if (rollback_num) // tokens
										{
											//printf(" deleting %d tokens\n", rollback_num);															
											embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());						
											n_past -= rollback_num;
											text_to_speak = utf8_substr(text_to_speak, 0, utf8_length(text_to_speak)-symbols_to_delete);
											last_output_needle = utf8_substr(last_output_needle, 0, utf8_length(last_output_needle)-symbols_to_delete);
											last_output_buffer = utf8_substr(last_output_buffer, 0, utf8_length(last_output_buffer)-symbols_to_delete);
											//printf("\nnew last_output_needle: (%s)\n", last_output_needle.c_str());
											temp_next = 1.8; // just for 1 token
											//continue;
										}
									}					
								}								
								// fill buffer for seqrep
								if (utf8_length(last_output_buffer) > 1000) last_output_buffer = utf8_substr(last_output_buffer, 100, last_output_buffer.size()-100); // in symbols not tokens, non-latin symbols takes 2x more			
								last_output_buffer += out_token_str;
							}
							
							if (text_to_speak == '\n'+params.person+':') 
							{
								person_name_is_found = 1;
								translation_is_going = 0;								
								//fprintf(stdout, " Found person name. translation_is_going: %d, text_to_speak: (%s)\n", translation_is_going, text_to_speak.c_str());
							}
							else if (text_to_speak[0] == '\n' && text_to_speak[text_to_speak.size()-1] == ':' && text_to_speak.size() < 10) // \nName:
							{
								bot_name_is_found = 1;
								bot_name_current = text_to_speak.substr(1, text_to_speak.size()-2);
								if (params.translate) bot_name_current_ru = translit_en_ru(bot_name_current);
								//fprintf(stdout, " Found bot name (%s). translation_is_going: %d, text_to_speak: (%s)\n", bot_name_current.c_str(), translation_is_going, text_to_speak.c_str());
								translation_full = "";
								text_to_speak = "";
							}
							//if (person_name_is_found) fprintf(stdout, " person_name_is_found, no tts: %d\n", person_name_is_found);
							//if (bot_name_is_found) fprintf(stdout, " bot_name_is_found %d\n", bot_name_is_found);
								
							int text_len = text_to_speak.size();
							if (text_to_speak[text_len-1] == ',') n_comas++;
							if (new_tokens == split_after && params.split_after && text_to_speak[text_len-1] == '\'') split_after++;
							if (text_to_speak.size() >= 3 && text_to_speak.substr(text_to_speak.size()-3, 3) == "Mr.") text_to_speak[text_len-1] = ' '; // no splitting on mr.
							
							// STOP on speech or hotkey for llama, every 2 tokens
							if (new_tokens % 2 == 0)
							{
								// check energy level, if user is speaking (it doesn't call whisper recognition, just a loud noise will stop everything)
								audio.get(2000, pcmf32_cur); // non-blocking, 2000 step_ms
								int vad_result = ::vad_simple_int(pcmf32_cur, WHISPER_SAMPLE_RATE, params.vad_last_ms, params.vad_thold, params.freq_thold, params.print_energy, params.vad_start_thold);
								
								if (!params.push_to_talk && vad_result == 1 || g_hotkey_pressed == "Ctrl+Space" || g_hotkey_pressed == "Alt") // speech started
								{
									llama_interrupted = 1;
									llama_interrupted_time = get_current_time_ms();
									printf(" [Speech/Stop!]\n");
									allow_xtts_file(params.xtts_control_path, 0); // xtts stop
									done = true; // llama generation stop
									g_hotkey_pressed = "";
									break;
								}								
							}
							//	clear mic
							if (new_tokens == 20 && !llama_interrupted)
							{
								audio.clear();
								//printf("\n [audio cleared after 20t]\n");
							}
							
							// splitting for tts
							if (text_len >= 2 && new_tokens >=2 && !person_name_is_found && ((new_tokens == split_after && params.split_after && text_to_speak[text_len-1] != '\'') || text_to_speak[text_len-1] == '.' || text_to_speak[text_len-1] == '(' || text_to_speak[text_len-1] == ')' || (text_to_speak[text_len-1] == ',' && n_comas==1 && new_tokens > split_after && params.split_after) || (text_to_speak[text_len-2] == ' ' && text_to_speak[text_len-1] == '-') ||  text_to_speak[text_len-1] == '?' || text_to_speak[text_len-1] == '!' || text_to_speak[text_len-1] == ';' || text_to_speak[text_len-1] == ':' || text_to_speak[text_len-1] == '\n'))
							{
								if (translation_is_going == 1) 
								{
									translation_full += text_to_speak;
									//fprintf(stdout, " translation_full: (%s)\n", translation_full.c_str());
								}
							
								//fprintf(stdout, " split_sign: (%c), translation_is_going: %d\n", text_to_speak[text_len-1], translation_is_going);
								text_to_speak = ::replace(text_to_speak, "\"", "'");
								text_to_speak = ::replace(text_to_speak, antiprompts[0], "");
								
								if (text_to_speak.size()) // first and mid parts of the sentence
								{
									// system TTS
									//int ret = system(("start /B "+params.speak + " " + std::to_string(voice_id) + " \"" + text_to_speak + "\" & exit").c_str()); // for windows
									//int ret = system((params.speak + " " + std::to_string(voice_id) + " \"" + text_to_speak + "\" &").c_str()); // for linux
									
									// translate
									// each generated sentence is translated by the same llama model, in the same ctx
									if (params.translate)
									{
										if (translation_is_going == 0)
										{
											std::string text_to_speak_translated = "";										
											n_embd_inp_before_trans = embd_inp.size();
											fprintf(stdout, "\n	Перевод:", n_embd_inp_before_trans);
											std::string trans_prompt = "\nПеревод последнего предложения на русский.\n"+bot_name_current_ru+":"+translation_full;
											//fprintf(stdout, "%s", trans_prompt.c_str());
											std::vector<llama_token> trans_prompt_emb = ::llama_tokenize(ctx_llama, trans_prompt, false); // prompt to tokens	
											embd.insert(embd.end(), trans_prompt_emb.begin(), trans_prompt_emb.end()); // inject prompt
											translation_is_going = 1; // started
											text_to_speak = "";
											//fprintf(stdout, " translation_is_going: 0->1\n");
											continue;
										}										
									}
									
									// XTTS in threads
									text_to_speak_arr[thread_i] = text_to_speak;
									reply_part_arr[thread_i] = reply_part;
									reply_part++;
									try 
									{
										threads.emplace_back([&] // creates a thread, threads are cleaned after user ends speech, after 80 threads
										{
											if (text_to_speak_arr[thread_i-1].size())
											{
												send_tts_async(text_to_speak_arr[thread_i-1], current_voice, params.language, params.xtts_url, reply_part_arr[thread_i-1]);
												text_to_speak_arr[thread_i-1] = "";
												reply_part_arr[thread_i-1] = 0;
											}
										});										
										thread_i++;
										text_to_speak = "";
										if (params.sleep_before_xtts) std::this_thread::sleep_for(std::chrono::milliseconds(params.sleep_before_xtts)); // 1s pause to speed up xtts/wav2lip inference
										
										// check energy level, if user is speaking (it doesn't call whisper recognition, just a loud noise will stop everything)
										if (!params.push_to_talk || params.push_to_talk && g_hotkey_pressed == "Alt")
										{
											audio.get(2000, pcmf32_cur); // non-blocking, 2000 step_ms
											int vad_result = ::vad_simple_int(pcmf32_cur, WHISPER_SAMPLE_RATE, params.vad_last_ms, params.vad_thold, params.freq_thold, params.print_energy, params.vad_start_thold);
											if (vad_result == 1) // speech started
											{
												llama_interrupted = 1;
												llama_interrupted_time = get_current_time_ms();
												printf(" [Speech!]\n");
												allow_xtts_file(params.xtts_control_path, 0); // xtts stop
												done = true; // llama generation stop
												break;
											}
										}
									}
									catch (const std::exception& ex) {
										std::cerr << "[Exception]: Failed to push_back mid thread: " << ex.what() << '\n';
									}
									
									// deleting translation from ctx
									if (params.translate && translation_is_going == 1)
									{										
										translation_is_going = 0; // finished
										//fprintf(stdout, " translation_is_going 1->0 \n");										
										
										if (n_embd_inp_before_trans && embd_inp.size()) 
										{
											int rollback_num = embd_inp.size()-n_embd_inp_before_trans;
											if (rollback_num)
											{														
												embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());						
												n_past -= rollback_num;
												//printf(" deleting %d tokens. embd_inp: %d \n", rollback_num, embd_inp.size());	
												printf("\n"); // to separate translation from original												
											}
											continue;
										}										
									}
								}
							}			   
                        }
                    }

                    {
						// TODO: check if llama_token_to_piece is fast enough
                        std::string last_output;
                        for (int i = embd_inp.size() - 10; i < (int) embd_inp.size(); i++) { // was 16
                            last_output += llama_token_to_piece(ctx_llama, embd_inp[i]);
                        }
                        last_output += llama_token_to_piece(ctx_llama, embd[0]);

						int i_antiprompt = 0;
						last_output_has_username = false;
						last_output_has_EOT = false;
                        for (std::string & antiprompt : antiprompts) 
						{
							// multiple character names for xtts
							if (params.multi_chars && last_output.size()>=4)
							{
								last_output = ::replace(last_output, " ???", ""); // there's a crash with regex, not sure why, todo 
								last_output = ::replace(last_output, " ??", "");
								last_output = ::replace(last_output, " ?", "");
								last_output = ::replace(last_output, " !!!", "");
								last_output = ::replace(last_output, " !!", "");
								last_output = ::replace(last_output, " !", "");
								last_output = ::replace(last_output, "!!!", "");
								last_output = ::replace(last_output, "!!", "");
								last_output = ::replace(last_output, " ...", "");
								last_output = ::replace(last_output, " .", "");
								last_output = ::replace(last_output, " ,", "");
								last_output = ::replace(last_output, "...", "");
								last_output = ::replace(last_output, "(", "");
								last_output = ::replace(last_output, ")", "");
								
								// new char found, will use its voice in Tts
								std::smatch matches;							
								std::regex r("\n([^:]*):", std::regex::icase | std::regex::optimize);
								if (std::regex_search(last_output, matches, r) && !matches.empty() && matches.size() >= 2 && !matches[1].str().empty() && matches[1].str() != params.person && matches[1].str() != " \n"+params.person) 
								{
									current_voice_tmp = matches[1].str();
									current_voice_tmp = ::replace(current_voice_tmp, ":", "");
									current_voice_tmp = ::replace(current_voice_tmp, "\"", "");
									trim(current_voice_tmp);
									if (current_voice_tmp.size()>1 && current_voice_tmp.size()<30) 
									{
										current_voice = current_voice_tmp;
										//printf(" new char found: (%s) %d s\n", current_voice, current_voice.size());
										std::regex regEx(current_voice+":");
										text_to_speak = std::regex_replace(text_to_speak, regEx, "");
									}									
								}
							}
							
							// stop words
                            if (last_output.length() > antiprompt.length() && last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) 
							{
                                done = true;
                                text_to_speak = ::replace(text_to_speak, antiprompt, "");
                                fflush(stdout);
                                need_to_save_session = true;
								if (i_antiprompt == 0) 
								{
									last_output_has_username = true;
									printf(" "); // for typed input
								}
								else if (i_antiprompt == eot_antiprompt_id_1 || i_antiprompt == eot_antiprompt_id_2) 
								{
									last_output_has_EOT = true;									
								}
								if (antiprompt == params.instruct_preset_data["bot_message_suffix"] || antiprompt == "</end_of_turn>" ) // delete printed tag
								{
									std::string backspaces(antiprompt.length(), '\b');
									std::string spaces(antiprompt.length(), ' ');
									fflush(stdout);
									printf("%s", backspaces); // remove printed tag
									printf("%s", spaces); // replace with spaces
									printf("%s", backspaces); // remove spaces (?!)
									printf("\n");
									fflush(stdout);
								}
									
								//printf(" antiprompt: (%s), t:%d\n", antiprompt.c_str(), tokens_in_reply);
								
								// min tokens in reply
								if (params.min_tokens && tokens_in_reply < params.min_tokens)
								{
									int symbols_to_delete = static_cast<int>(utf8_length(antiprompt) * 1) + 1; // +\n
									const std::vector<llama_token> tokens_to_del = llama_tokenize(ctx_llama, antiprompt.c_str(), false);
									int rollback_num = tokens_to_del.size() + 1; // + \n
									if (rollback_num) // tokens
									{										
										embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
										//printf(" deleting %d tokens. Tokens in ctx: %d\n", rollback_num, embd_inp.size());
										n_past -= rollback_num;
										if (symbols_to_delete > utf8_length(text_to_speak)) text_to_speak = "";
										else text_to_speak = utf8_substr(text_to_speak, 0, utf8_length(text_to_speak)-symbols_to_delete);
										temp_next = 1.8; // just for 1 token
										
										fflush(stdout);
										printf("\b\b\b\b\b\b\b\b\b\b\b\b"); // remove printed person_name
										fflush(stdout);
										done = false;
									}
									if (params.debug)
									{
										// print all emb for debug
										std::string full_dialog = emb_to_str(ctx_llama, embd_inp);
										printf("\n=====FULL text in embd (%d tokens, %d symbols)=====\n%s\n====END====\n", embd_inp.size(), full_dialog.size(), full_dialog.c_str());
									}
								}
								else 
								{		
									if (params.debug)
									{
										// print all emb for debug
										std::string full_dialog = emb_to_str(ctx_llama, embd_inp);
										printf("\n=====FULL text in embd (%d tokens, %d symbols)=====\n%s\n====END====\n", embd_inp.size(), full_dialog.size(), full_dialog.c_str());
									}
									break;
								}
                            }
							i_antiprompt++;
                        }
                    }
					

                    is_running = sdl_poll_events();

                    if (!is_running) {
                        break;
                    }
                }

				// final part of the sentence, if any
                text_to_speak = ::replace(text_to_speak, "\"", "'");				
				if (text_to_speak.size()) 
				{
					text_to_speak_arr[thread_i] = text_to_speak;
					reply_part_arr[thread_i] = reply_part;
					reply_part++;
					try 
					{		
						threads.emplace_back([&] // creates and starts a thread
						{
							if (text_to_speak_arr[thread_i-1].size())
							{
								send_tts_async(text_to_speak_arr[thread_i-1], current_voice, params.language, params.xtts_url, reply_part_arr[thread_i-1]);
								text_to_speak_arr[thread_i-1] = "";
								reply_part_arr[thread_i-1] = 0;
							}
						});	
						thread_i++;						
						text_to_speak = "";  						
					}
					catch (const std::exception& ex) {
						std::cerr << "[Exception]: Failed to emplace fin thread: " << ex.what() << '\n'; 
					}
				}
				//if ((embd_inp.size() % 10) == 0) printf("\n [t: %zu]\n", embd_inp.size());
                
				if (llama_interrupted /*&& llama_interrupted_time - llama_start_time < 2.0*/)
				{
					1;
					//printf(" \n[continue speech] (%f)", (llama_interrupted_time - llama_start_time));
				}
				else 
				{
					audio.clear();
					//printf("\n [audio cleared fin]\n");
				}
				
				
				llama_end_time = get_current_time_ms();
				if (params.verbose) 
				{
					llama_time_input = llama_start_generation_time - llama_start_time;
					llama_time_output = llama_end_time - llama_start_generation_time;
					llama_time_total = llama_end_time - llama_start_time;
					printf("\n\n[tokens: %d in + %d out. Input %.3f s + output %.3f s = total: %.3f s]", input_tokens_count , new_tokens, llama_time_input, llama_time_output, llama_time_total);
					printf("\n[Speed: input %.2f t/s + output %.2f t/s = total: %.2f t/s]\n", input_tokens_count/llama_time_input, new_tokens/llama_time_output, new_tokens/llama_time_total );
				}
				llama_interrupted = 0;
				llama_interrupted_time = 0.0;
				llama_start_generation_time = 0.0;
				g_hotkey_pressed = "";
            }
        }
    }

    audio.pause();

    whisper_print_timings(ctx_wsp);
    whisper_free(ctx_wsp);

    llama_print_timings(ctx_llama);
    llama_free(ctx_llama);
	
	input_thread.join();    // kb input
	shortcut_thread.join(); // shortcuts

    return 0;
}

	
#if _WIN32
int wmain(int argc, const wchar_t ** argv_UTF16LE) {
	console::init(true, true);
    atexit([]() { console::cleanup(); });
    std::vector<std::string> buffer(argc);
    std::vector<const char*> argv_UTF8(argc);
	for (int i = 0; i < argc; ++i) {
        buffer[i] = console::UTF16toUTF8(argv_UTF16LE[i]);
        argv_UTF8[i] = buffer[i].c_str();
    }
    return run(argc, argv_UTF8.data());
}
#else
int main(int argc, const char ** argv_UTF8) {
    console::init(true, true);
    atexit([]() { console::cleanup(); });
    return run(argc, argv_UTF8);
}
#endif

