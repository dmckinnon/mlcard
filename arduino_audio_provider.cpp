/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
//#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include "audio_provider.h"

#include <stdio.h>
#include <string.h>

#include "pico/stdlib.h"

#include "pico/pdm_microphone.h"
#include "micro_features_micro_model_settings.h"

namespace {
constexpr int DEFAULT_PDM_BUFFER_SIZE = kMaxAudioSampleSize;

bool g_is_audio_initialized = false;
// An internal buffer able to fit 16x our sample size
constexpr int kAudioCaptureBufferSize = DEFAULT_PDM_BUFFER_SIZE * 16;
// configuration
const struct pdm_microphone_config config = {
    // GPIO pin for the PDM DAT signal
    .gpio_data = 2,

    // GPIO pin for the PDM CLK signal
    .gpio_clk = 3,

    // PIO instance to use
    .pio = pio0,

    // PIO State Machine instance to use
    .pio_sm = 0,

    // sample rate in Hz
    .sample_rate = kAudioSampleFrequency,

    // number of samples to buffer
    .sample_buffer_size = DEFAULT_PDM_BUFFER_SIZE,
};


// Add an extra capture amount due to overrun.
// The index gets up to the end but doesn't have a wraparound function inside
// the audio capture. I could write one, or I could do this.
int16_t g_audio_capture_buffer[kAudioCaptureBufferSize+DEFAULT_PDM_BUFFER_SIZE];
// A buffer that holds our output
int16_t g_audio_output_buffer[kMaxAudioSampleSize];
// Mark as volatile so we can check in a while loop to see if
// any samples have arrived yet.
volatile int32_t g_latest_audio_timestamp = 0;
}  // namespace

void CaptureSamples() {
  // This is how many bytes of new data we have each time this is called
  const int number_of_samples = DEFAULT_PDM_BUFFER_SIZE/2;
  // Calculate what timestamp the last audio sample represents
  const int32_t time_in_ms =
      g_latest_audio_timestamp +
      (number_of_samples / (kAudioSampleFrequency / 1000));
  // Determine the index, in the history of all samples, of the last sample
  const int32_t start_sample_offset =
      g_latest_audio_timestamp * (kAudioSampleFrequency / 1000);
  // Determine the index of this sample in our ring buffer
  const int capture_index = start_sample_offset % kAudioCaptureBufferSize;

  pdm_microphone_read(g_audio_capture_buffer + capture_index, DEFAULT_PDM_BUFFER_SIZE);
  // This is how we let the outside world know that new audio data has arrived.
  g_latest_audio_timestamp = time_in_ms;
}

TfLiteStatus InitAudioRecording(tflite::ErrorReporter* error_reporter) {

  if (pdm_microphone_init(&config) < 0) {
    printf("PDM microphone initialization failed! Infinite loop\n");
    while (1) { tight_loop_contents(); }
  }

  // Hook up the callback that will be called with each sample
  pdm_microphone_set_samples_ready_handler(CaptureSamples);
  pdm_microphone_set_filter_gain(20);

  // Start listening for audio: MONO @ 16KHz with gain at 20
  if (pdm_microphone_start() < 0) {
    printf("PDM microphone start failed! Infinite loop\n");
    while (1) { tight_loop_contents(); }
  }

  // Block until we have our first audio sample
  while (!g_latest_audio_timestamp) {
    printf("blocking on audio\n");
  }

  //printf("----------------------------------\n\n\n\n------------------------------------\n");
  //printf("LET THE AUDIO BEGIN\n");

  return kTfLiteOk;
}

TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  //printf("Get audio from time %d for %d ms\n", start_ms, duration_ms);
  // Set everything up to start receiving audio
  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording(error_reporter);
    if (init_status != kTfLiteOk) {
      printf("audio failed to init\n");
      return init_status;
    }
    g_is_audio_initialized = true;
  }
  // This next part should only be called when the main thread notices that the
  // latest audio sample data timestamp has changed, so that there's new data
  // in the capture ring buffer. The ring buffer will eventually wrap around and
  // overwrite the data, but the assumption is that the main thread is checking
  // often enough and the buffer is large enough that this call will be made
  // before that happens.

  // Determine the index, in the history of all samples, of the first sample we want
  const int start_offset = start_ms * (kAudioSampleFrequency / 1000);
  // Determine how many samples we want in total
  const int duration_sample_count =
      duration_ms * (kAudioSampleFrequency / 1000);

  for (int i = 0; i < duration_sample_count; ++i) {
    // For each sample, transform its index in the history of all samples into
    // its index in g_audio_capture_buffer
    const int capture_index = (start_offset + i) % kAudioCaptureBufferSize;
    // Write the sample to the output buffer
    g_audio_output_buffer[i] = g_audio_capture_buffer[capture_index];
  }

  // Set pointers to provide access to the audio
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;

  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() { return g_latest_audio_timestamp; }

#endif  // ARDUINO_EXCLUDE_CODE
