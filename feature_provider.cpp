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

#include "feature_provider.h"
#include <cstdio>
#include "audio_provider.h"
#include "micro_features_micro_features_generator.h"
#include "micro_features_micro_model_settings.h"

FeatureProvider::FeatureProvider(int feature_size, int8_t* feature_data)
    : feature_size_(feature_size),
      feature_data_(feature_data),
      is_first_run_(true) {
  // Initialize the feature data to default values.

  


  for (int n = 0; n < feature_size_; ++n) {
    feature_data_[n] = 0;
  }
}

FeatureProvider::~FeatureProvider() {}

TfLiteStatus FeatureProvider::PopulateFeatureData(
    tflite::ErrorReporter* error_reporter, int32_t last_time_in_ms,
    int32_t time_in_ms, int* how_many_new_slices) {
  if (feature_size_ != kFeatureElementCount) {
    printf("Requested feature_data_ size %d doesn't match %d\n",
                         feature_size_, kFeatureElementCount);
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Requested feature_data_ size %d doesn't match %d",
                         feature_size_, kFeatureElementCount);
    return kTfLiteError;
  }

  // Quantize the time into steps as long as each window stride, so we can
  // figure out which audio data we need to fetch.
  //printf("Figuring out steps, %d / %d\n", time_in_ms, kFeatureSliceStrideMs);
  const int last_step = (last_time_in_ms / kFeatureSliceStrideMs);
  const int current_step = (time_in_ms / kFeatureSliceStrideMs);
  printf("time window: %d - %d = %d\n", current_step, last_step, current_step - last_step);

  int slices_needed = current_step - last_step;
  // If this is the first call, make sure we don't use any cached information.
  if (is_first_run_) {
    printf("Init features\n");
    TfLiteStatus init_status = InitializeMicroFeatures(error_reporter);
    if (init_status != kTfLiteOk) {
      printf("Failed feature init\n");
      return init_status;
    }
    is_first_run_ = false;
    slices_needed = kFeatureSliceCount;
  }
  if (slices_needed > kFeatureSliceCount) {
    slices_needed = kFeatureSliceCount;
  }
  *how_many_new_slices = slices_needed;

  const int slices_to_keep = kFeatureSliceCount - slices_needed;
  const int slices_to_drop = kFeatureSliceCount - slices_to_keep;
  //printf("keeping %d slices of total %d\n", slices_to_keep, kFeatureSliceCount);
  // If we can avoid recalculating some slices, just move the existing data
  // up in the spectrogram, to perform something like this:
  // last time = 80ms          current time = 120ms
  // +-----------+             +-----------+
  // | data@20ms |         --> | data@60ms |
  // +-----------+       --    +-----------+
  // | data@40ms |     --  --> | data@80ms |
  // +-----------+   --  --    +-----------+
  // | data@60ms | --  --      |  <empty>  |
  // +-----------+   --        +-----------+
  // | data@80ms | --          |  <empty>  |
  // +-----------+             +-----------+
  if (slices_to_keep > 0) {
    for (int dest_slice = 0; dest_slice < slices_to_keep; ++dest_slice) {
      int8_t* dest_slice_data =
          feature_data_ + (dest_slice * kFeatureSliceSize);
      const int src_slice = dest_slice + slices_to_drop;
      const int8_t* src_slice_data =
          feature_data_ + (src_slice * kFeatureSliceSize);
      for (int i = 0; i < kFeatureSliceSize; ++i) {
        dest_slice_data[i] = src_slice_data[i];
      }
    }
  }
  //printf("Pull audio for %d slices\n", slices_needed);
  // Any slices that need to be filled in with feature data have their
  // appropriate audio data pulled, and features calculated for that slice.
  int numFeatureSetsGenerated = 0;
  if (slices_needed > 0) {
    printf(
      "Feature gen stats: \nto keep: %d\nkFeatureSliceCount: %d\nslices needed: %d\n",
      slices_to_keep, kFeatureSliceCount, slices_needed);
    for (int new_slice = slices_to_keep; new_slice < kFeatureSliceCount;
         ++new_slice) {
      const int new_step = (current_step - kFeatureSliceCount + 1) + new_slice;
      const int32_t slice_start_ms = (new_step * kFeatureSliceStrideMs);
      int16_t* audio_samples = nullptr;
      int audio_samples_size = 0;
      // TODO(petewarden): Fix bug that leads to non-zero slice_start_ms
      // I'm hitting this bug. Why is this a problem? oh, negative
      GetAudioSamples(error_reporter, (slice_start_ms > 0 ? slice_start_ms : 0),
                      kFeatureSliceDurationMs, &audio_samples_size,
                      &audio_samples);
      if (audio_samples_size < kMaxAudioSampleSize) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Audio data size %d too small, want %d",
                             audio_samples_size, kMaxAudioSampleSize);
        return kTfLiteError;
      }
      int8_t* new_slice_data = feature_data_ + (new_slice * kFeatureSliceSize);
      size_t num_samples_read;
      //rintf("Generating micro features\n");
      TfLiteStatus generate_status = GenerateMicroFeatures(
          error_reporter, audio_samples, audio_samples_size, kFeatureSliceSize,
          new_slice_data, &num_samples_read);
      numFeatureSetsGenerated ++;
      if (generate_status != kTfLiteOk) {
        printf("Failed generating features\n");
        return generate_status;
      }
      //printf("Finsihed generating features\n");
    }
  }
  if (numFeatureSetsGenerated != 0)
    printf("Generated %d feature sets\n", numFeatureSetsGenerated);
  return kTfLiteOk;
}
