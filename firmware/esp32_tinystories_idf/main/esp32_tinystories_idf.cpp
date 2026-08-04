// ESP-IDF port of the released TinyStories PLE runtime. Board Manager owns
// board/display initialization, and the model stays mapped from flash.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_board_manager.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"

#define LLM_INT8_ACT 1
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#include "llm.h"
#include "vocab.h"

#include "board_display.h"

namespace {

constexpr int kPromptIds[] = {433, 447, 259, 405}; // "Once upon a time"
constexpr int kGenerateCount = 200;
constexpr int kParallelRowThreshold = 128;
constexpr size_t kStaticSramBytes = 2 * LLM_Q8_MAX_INPUT;
const char *const TAG = "esp32_tinystories_idf";

Model model;
Scratch scratch;

const uint8_t *model_image;
esp_partition_mmap_handle_t model_mapping;
size_t psram_used;
size_t sram_used;

TaskHandle_t matvec_worker;
TaskHandle_t inference_task;
const QT *matvec_job_tensor;
const int8_t *matvec_job_input;
float matvec_job_input_scale;
float *matvec_job_output;
int matvec_job_split;

void *alloc_psram(size_t size) {
    void *memory = heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (memory) {
        psram_used += size;
    }
    return memory;
}

void *alloc_required_psram(size_t size, const char *name) {
    void *memory = alloc_psram(size);
    if (!memory) {
        ESP_LOGE(TAG, "required PSRAM allocation failed: %s (%u bytes)", name,
                 static_cast<unsigned>(size));
    }
    return memory;
}

void *alloc_required_sram(size_t size, const char *name) {
    void *memory = heap_caps_malloc(size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!memory) {
        ESP_LOGE(TAG, "required SRAM allocation failed: %s (%u bytes)", name,
                 static_cast<unsigned>(size));
        return nullptr;
    }
    sram_used += size;
    return memory;
}

esp_err_t allocate_sram_buffer(float **buffer, size_t elements,
                               const char *name) {
    *buffer =
        static_cast<float *>(alloc_required_sram(elements * sizeof(float), name));
    return *buffer ? ESP_OK : ESP_ERR_NO_MEM;
}

esp_err_t allocate_psram_buffer(float **buffer, size_t elements,
                                const char *name) {
    *buffer = static_cast<float *>(
                  alloc_required_psram(elements * sizeof(float), name));
    return *buffer ? ESP_OK : ESP_ERR_NO_MEM;
}

void matvec_worker_main(void *) {
    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        matvec_i8_range(matvec_job_tensor, matvec_job_input, matvec_job_input_scale,
                        matvec_job_output, 0, matvec_job_split);
        xTaskNotifyGive(inference_task);
    }
}

void matvec_parallel(const QT *tensor, const float *input, float *output) {
    static int8_t quantized_input[LLM_Q8_MAX_INPUT];
    if (!matvec_worker || !tensor->w8 || tensor->rows < kParallelRowThreshold) {
        MATVEC(tensor, input, output);
        return;
    }

    float input_scale;
    quantize_act(input, tensor->cols, quantized_input, &input_scale);
    matvec_job_tensor = tensor;
    matvec_job_input = quantized_input;
    matvec_job_input_scale = input_scale;
    matvec_job_output = output;
    matvec_job_split = tensor->rows / 2;
    xTaskNotifyGive(matvec_worker);
    matvec_i8_range(tensor, quantized_input, input_scale, output,
                    matvec_job_split, tensor->rows);
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

esp_err_t allocate_scratch() {
    const Cfg &config = model.c;
    const size_t dimension = static_cast<size_t>(config.dim);
    const size_t layers = static_cast<size_t>(config.n_layers);
    const size_t ple_dimension = static_cast<size_t>(config.ple_dim);
    const size_t ffn_dimension = static_cast<size_t>(config.ffn);
    const size_t sequence = static_cast<size_t>(config.seq_len);

    ESP_RETURN_ON_ERROR(allocate_sram_buffer(&scratch.x, dimension, "x"), TAG,
                        "allocate x failed");
    ESP_RETURN_ON_ERROR(allocate_sram_buffer(
                            &scratch.h,
                            ffn_dimension > dimension ? ffn_dimension : dimension,
                            "h"),
                        TAG, "allocate h failed");
    ESP_RETURN_ON_ERROR(allocate_sram_buffer(&scratch.qkv, 3 * dimension, "qkv"),
                        TAG, "allocate qkv failed");
    ESP_RETURN_ON_ERROR(allocate_sram_buffer(&scratch.att, dimension, "att"), TAG,
                        "allocate att failed");
    ESP_RETURN_ON_ERROR(allocate_sram_buffer(&scratch.g1, ffn_dimension, "g1"),
                        TAG, "allocate g1 failed");
    ESP_RETURN_ON_ERROR(
        allocate_sram_buffer(
            &scratch.g2,
            ple_dimension > ffn_dimension ? ple_dimension : ffn_dimension, "g2"),
        TAG, "allocate g2 failed");
    ESP_RETURN_ON_ERROR(
        allocate_sram_buffer(&scratch.ple, layers * ple_dimension, "ple"), TAG,
        "allocate ple failed");
    ESP_RETURN_ON_ERROR(
        allocate_sram_buffer(&scratch.tmpP, layers * ple_dimension, "tmpP"), TAG,
        "allocate tmpP failed");
    ESP_RETURN_ON_ERROR(
        allocate_sram_buffer(&scratch.trow, layers * ple_dimension, "trow"), TAG,
        "allocate trow failed");
    ESP_RETURN_ON_ERROR(allocate_sram_buffer(&scratch.scores, sequence, "scores"),
                        TAG, "allocate scores failed");

    ESP_RETURN_ON_ERROR(
        allocate_psram_buffer(&scratch.logits,
                              static_cast<size_t>(model.out_vocab), "logits"),
        TAG, "allocate logits failed");
    ESP_RETURN_ON_ERROR(allocate_psram_buffer(&scratch.kcache,
                        layers * sequence * dimension,
                        "kcache"),
                        TAG, "allocate kcache failed");
    ESP_RETURN_ON_ERROR(allocate_psram_buffer(&scratch.vcache,
                        layers * sequence * dimension,
                        "vcache"),
                        TAG, "allocate vcache failed");
    return ESP_OK;
}

esp_err_t copy_norms_to_sram() {
    const int dimension = model.c.dim;
    const int layers = model.c.n_layers;
    const int ple_dimension = model.c.ple_dim;
    const float **vectors[3 * LLM_MAX_LAYERS + 2];
    int sizes[3 * LLM_MAX_LAYERS + 2];
    int vector_count = 0;

    vectors[vector_count] = &model.ple_proj_norm;
    sizes[vector_count++] = ple_dimension;
    for (int layer = 0; layer < layers; ++layer) {
        vectors[vector_count] = &model.attn_norm[layer];
        sizes[vector_count++] = dimension;
        vectors[vector_count] = &model.ffn_norm[layer];
        sizes[vector_count++] = dimension;
        vectors[vector_count] = &model.ple_norm[layer];
        sizes[vector_count++] = dimension;
    }
    vectors[vector_count] = &model.out_norm;
    sizes[vector_count++] = dimension;

    for (int index = 0; index < vector_count; ++index) {
        const size_t bytes = static_cast<size_t>(sizes[index]) * sizeof(float);
        void *destination = alloc_required_sram(bytes, "norm vector");
        ESP_RETURN_ON_FALSE(destination, ESP_ERR_NO_MEM, TAG,
                            "copy norm vector failed");
        memcpy(destination, *vectors[index], bytes);
        *vectors[index] = static_cast<const float *>(destination);
    }

    ESP_LOGI(TAG, "norms -> SRAM: %d vectors", vector_count);
    return ESP_OK;
}

esp_err_t stage_weights_to_psram() {
    const int expected = llm_core_stage_count(&model);
    int staged = llm_stage_core_int8_alloc(&model, alloc_psram);
    ESP_RETURN_ON_FALSE(staged == expected, ESP_ERR_NO_MEM, TAG,
                        "staged only %d/%d core tensors", staged, expected);

    void *head_buffer = alloc_required_psram(
                            llm_stage_int8_bytes(&model.out_head), "output head");
    ESP_RETURN_ON_FALSE(head_buffer, ESP_ERR_NO_MEM, TAG,
                        "stage output head failed");
    llm_stage_int8(&model.out_head, head_buffer);
    ++staged;

    ESP_LOGI(TAG, "weights -> PSRAM: %d tensors, %.2f MB allocated", staged,
             psram_used / 1048576.0);
    return ESP_OK;
}

void initialize_parallel_matvec() {
#if CONFIG_FREERTOS_NUMBER_OF_CORES > 1
    inference_task = xTaskGetCurrentTaskHandle();
    const int inference_core = xPortGetCoreID();
    const int worker_core =
        (inference_core + 1) % CONFIG_FREERTOS_NUMBER_OF_CORES;
    const BaseType_t created =
        xTaskCreatePinnedToCore(matvec_worker_main, "llm_matvec", 4096, nullptr,
                                2, &matvec_worker, worker_core);
    if (created == pdPASS) {
        model.layer_matvec = matvec_parallel;
        model.head_matvec = matvec_parallel;
        ESP_LOGI(TAG, "inference core=%d, matvec worker core=%d", inference_core,
                 worker_core);
    } else {
        ESP_LOGW(TAG, "matvec worker creation failed; using one core");
    }
#else
    ESP_LOGI(TAG, "single-core target; using one-core int8 matvec");
#endif
}

uint32_t model_fingerprint() {
    uint32_t fingerprint = 2166136261U;
    for (size_t index = 0; index < model.image_bytes; ++index) {
        fingerprint ^= model_image[index];
        fingerprint *= 16777619U;
    }
    return fingerprint;
}

esp_err_t load_model_partition() {
    const esp_partition_t *partition = esp_partition_find_first(
                                           ESP_PARTITION_TYPE_DATA, static_cast<esp_partition_subtype_t>(0x40),
                                           "model");
    ESP_RETURN_ON_FALSE(partition, ESP_ERR_NOT_FOUND, TAG,
                        "model partition not found");

    const void *mapped_model = nullptr;
    ESP_RETURN_ON_ERROR(esp_partition_mmap(partition, 0, partition->size,
                                           ESP_PARTITION_MMAP_DATA, &mapped_model,
                                           &model_mapping),
                        TAG, "map model partition failed");
    model_image = static_cast<const uint8_t *>(mapped_model);

    const int load_result = llm_load(model_image, &model);
    ESP_RETURN_ON_FALSE(load_result == 0, ESP_ERR_INVALID_RESPONSE, TAG,
                        "invalid model image (llm_load=%d)", load_result);
    ESP_RETURN_ON_FALSE(model.image_bytes <= partition->size,
                        ESP_ERR_INVALID_SIZE, TAG,
                        "model image exceeds its partition");
    ESP_RETURN_ON_FALSE(VOCAB_N == model.out_vocab, ESP_ERR_INVALID_SIZE, TAG,
                        "tokenizer/model mismatch: vocab.h=%d, model=%d", VOCAB_N,
                        model.out_vocab);

    const Cfg &config = model.c;
    ESP_LOGI(TAG,
             "model: Vin=%d Vout=%d D=%d L=%d H=%d F=%d P=%d (image %.2f MB)",
             config.vocab, model.out_vocab, config.dim, config.n_layers,
             config.n_heads, config.ffn, config.ple_dim,
             model.image_bytes / 1048576.0);
    return ESP_OK;
}

esp_err_t initialize_runtime() {
    ESP_RETURN_ON_ERROR(allocate_scratch(), TAG, "scratch allocation failed");
    ESP_RETURN_ON_ERROR(copy_norms_to_sram(), TAG, "norm relocation failed");
    ESP_LOGI(TAG, "hot set -> SRAM: %u B dynamic + %u B static = %u B managed",
             static_cast<unsigned>(sram_used),
             static_cast<unsigned>(kStaticSramBytes),
             static_cast<unsigned>(sram_used + kStaticSramBytes));

    ESP_RETURN_ON_ERROR(stage_weights_to_psram(), TAG, "weight staging failed");
    initialize_parallel_matvec();

    ESP_LOGI(TAG, "build: bytes=%u fp=%08x sram=%uB psram=%.2fMB",
             static_cast<unsigned>(model.image_bytes),
             static_cast<unsigned>(model_fingerprint()),
             static_cast<unsigned>(sram_used + kStaticSramBytes),
             psram_used / 1048576.0);
    ESP_LOGI(TAG, "free: SRAM %.0f KB, PSRAM %.2f MB",
             heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0,
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1048576.0);
    return ESP_OK;
}

void emit_token(int token) {
    if (token < 0 || token >= VOCAB_N) {
        return;
    }
    const uint8_t *bytes = VOCAB_BLOB + VOCAB_OFF[token];
    const int length = VOCAB_OFF[token + 1] - VOCAB_OFF[token];
    fwrite(bytes, 1, static_cast<size_t>(length), stdout);
    fflush(stdout);
    board_display_puts(bytes, static_cast<size_t>(length));
}

void generate_story() {
    printf("\n>>> ");
    int position = 0;
    int token = 0;

    for (const int prompt_token : kPromptIds) {
        token = prompt_token;
        emit_token(token);
        llm_forward(&model, token, position++, &scratch);
    }

    llm_profile_reset(&scratch);
    const int64_t start = esp_timer_get_time();
    int64_t decode_microseconds = 0;
    int decoded = 0;

    for (int step = 0; step < kGenerateCount && position < model.c.seq_len;
            ++step) {
        int best = 0;
        float best_value = -1e30f;
        for (int vocabulary_index = 0; vocabulary_index < model.out_vocab;
                ++vocabulary_index) {
            if (scratch.logits[vocabulary_index] > best_value) {
                best_value = scratch.logits[vocabulary_index];
                best = vocabulary_index;
            }
        }

        token = best;
        emit_token(token);
        const int64_t decode_start = esp_timer_get_time();
        llm_forward(&model, token, position++, &scratch);
        decode_microseconds += esp_timer_get_time() - decode_start;
        ++decoded;
        if ((step & 7) == 0) {
            vTaskDelay(1);
        }
    }

    if (decoded == 0 || decode_microseconds == 0) {
        ESP_LOGW(TAG, "no tokens decoded");
        return;
    }

    const int64_t total_microseconds = esp_timer_get_time() - start;
    const float total_tokens_per_second =
        decoded * 1e6f / static_cast<float>(total_microseconds);
    const float milliseconds_per_token = decode_microseconds / 1000.0f / decoded;
    printf("\n\n--- %d tokens in %.2f s ---\n", decoded,
           total_microseconds / 1e6);
    printf("throughput: %.2f tok/s (%.1f ms/token compute)\n",
           total_tokens_per_second, milliseconds_per_token);

    if (scratch.profile.calls) {
        const float divisor = static_cast<float>(scratch.profile.calls) * 1000.0f;
        printf("profile ms/token: input %.1f | attn %.1f | ffn %.1f | ple %.1f | "
               "head %.1f\n",
               scratch.profile.input_us / divisor,
               scratch.profile.attn_us / divisor, scratch.profile.ffn_us / divisor,
               scratch.profile.ple_us / divisor, scratch.profile.head_us / divisor);
    }
    board_display_stats(decoded * 1e6f / decode_microseconds,
                        milliseconds_per_token);
}

void show_error(const char *message) {
    board_display_puts(reinterpret_cast<const uint8_t *>(message),
                       strlen(message));
}

} // namespace

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "TinyStories PLE runtime (ESP-IDF + Board Manager)");
    ESP_ERROR_CHECK(esp_board_manager_print_board_info());

    const size_t psram_total = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "PSRAM available: %.2f MB", psram_total / 1048576.0);
    if (psram_total == 0) {
        ESP_LOGE(TAG, "PSRAM is required");
        return;
    }

    ESP_ERROR_CHECK(board_display_init());
    const esp_err_t model_error = load_model_partition();
    if (model_error != ESP_OK) {
        show_error(
            "\nMODEL PARTITION INVALID\nFLASH TINYSTORIES MODEL.BIN AND RESTART\n");
        ESP_LOGE(TAG, "model is unavailable: %s", esp_err_to_name(model_error));
        return;
    }

    const esp_err_t runtime_error = initialize_runtime();
    if (runtime_error != ESP_OK) {
        show_error("\nRUNTIME INITIALIZATION FAILED\nCHECK SRAM AND PSRAM\n");
        ESP_LOGE(TAG, "runtime initialization failed: %s",
                 esp_err_to_name(runtime_error));
        return;
    }

    generate_story();
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
