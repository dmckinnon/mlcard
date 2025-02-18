/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "command_responder.h"
#include "pico/stdlib.h"
#include <cstdio>

//#include "Arduino.h"

#define LEDR LED4
#define LEDB LED4
#define LEDG LED4

const uint8_t LED_PIN = 25;

// Toggles the built-in LED every inference, and lights a colored LED depending
// on which word was detected.
void RespondToCommand(tflite::ErrorReporter* error_reporter,
                      int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
  static bool is_initialized = false;
  //printf("Respond to Command\n");
  if (!is_initialized) {
    //pinMode(LED_BUILTIN, OUTPUT);
    // Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
    //pinMode(LEDR, OUTPUT);
    //pinMode(LEDG, OUTPUT);
    //pinMode(LEDB, OUTPUT);
    // Ensure the LED is off by default.
    // Note: The RGB LEDs on the Arduino Nano 33 BLE
    // Sense are on when the pin is LOW, off when HIGH.
    //digitalWrite(LEDR, HIGH);
    //digitalWrite(LEDG, HIGH);
    //digitalWrite(LEDB, HIGH);
    
    is_initialized = true;
  }
  static int32_t last_command_time = 0;
  static int count = 0;
  static int certainty = 220;

  if (score < 100)
  {
    return;
  }

  if (score > 120 && current_time - last_command_time > 700
      && found_command != "unknown" && found_command != "silence") {
    //TF_LITE_REPORT_ERROR(error_reporter, "Successful inference, heard %s (%d) @%dms", found_command,
    //                     score, current_time);
    printf("Successful inference, heard %s (%d) @%dms\n", found_command,
                         score, current_time);
    last_command_time = current_time;
    // If we hear a command, light up the LED
    
    
    // Run a loop for each number
    // Right now turn them all on
    gpio_put(16, 1);
    gpio_put(17, 1);
    gpio_put(18, 1);
    gpio_put(19, 1);
    gpio_put(20, 0);
    gpio_put(21, 0);
    sleep_ms(1000);
  }

  // If last_command_time is non-zero but was >3 seconds ago, zero it
  // and switch off the LED.
  if (last_command_time != 0) {
    if (last_command_time < (current_time - 1000)) {
      last_command_time = 0;
      
      printf("Zeroing\n");
      gpio_put(17, 0);
      for (int i = 16; i < 22; ++i)
      {
        gpio_put(i, 0);
      }
    }
  }

  // Otherwise, toggle the LED every time an inference is performed.
  ++count;
  if (count & 1) {
    //Serial.println("Half inference");
    printf("General inference, heard %s (%d) @%dms\n", found_command,
                        score, current_time);
    gpio_put(15, 1);
  } else {
    printf("General inference, heard %s (%d) @%dms\n", found_command,
                         score, current_time);
    gpio_put(15, 0);
  }
}

#endif  // ARDUINO_EXCLUDE_CODE
