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

void RunLEDs(const char* value)
{
  
  if (value == "one")
  {
    for (int i = 0; i < 100; ++i)
    {
      gpio_put(20, 1);
      gpio_put(21, 0);
      gpio_put(19, 1);
      gpio_put(18, 1);
      gpio_put(17, 0);
      gpio_put(16, 1);
      sleep_ms(5);
      gpio_put(20, 0);
      gpio_put(21, 1);
      gpio_put(19, 0);
      gpio_put(18, 0);
      gpio_put(17, 0);
      gpio_put(16, 1);
      sleep_ms(5);
    }
  }
  else if (value == "two")
  {
    for (int i = 0; i < 100; ++i)
    {
      gpio_put(20, 0);
      gpio_put(21, 1);
      gpio_put(19, 1);
      gpio_put(18, 0);
      gpio_put(17, 1);
      gpio_put(16, 1);
      sleep_ms(5);
      gpio_put(20, 1);
      gpio_put(21, 0);
      gpio_put(19, 1);
      gpio_put(18, 1);
      gpio_put(17, 0);
      gpio_put(16, 1);
      sleep_ms(5);
    }
  }
  else if (value == "three")
  {
    for (int i = 0; i < 100; ++i)
    {
      gpio_put(20, 0);
      gpio_put(21, 1);
      gpio_put(19, 1);
      gpio_put(18, 1);
      gpio_put(17, 0);
      gpio_put(16, 1);
      sleep_ms(5);
      gpio_put(20, 1);
      gpio_put(21, 0);
      gpio_put(19, 1);
      gpio_put(18, 1);
      gpio_put(17, 0);
      gpio_put(16, 1);
      sleep_ms(5);
    }
  }
  else if (value == "four")
  {
    for (int i = 0; i < 100; ++i)
    {
      gpio_put(20, 0);
      gpio_put(21, 1);
      gpio_put(19, 1);
      gpio_put(18, 1);
      gpio_put(17, 0);
      gpio_put(16, 0);
      sleep_ms(5);
      gpio_put(20, 1);
      gpio_put(21, 0);
      gpio_put(19, 0);
      gpio_put(18, 1);
      gpio_put(17, 1);
      gpio_put(16, 1);
      sleep_ms(5);
    }
  }
  else if (value == "five")
  {
    for (int i = 0; i < 100; ++i)
    {
      gpio_put(20, 0);
      gpio_put(21, 1);
      gpio_put(19, 0);
      gpio_put(18, 1);
      gpio_put(17, 0);
      gpio_put(16, 1);
      sleep_ms(5);
      gpio_put(20, 1);
      gpio_put(21, 0);
      gpio_put(19, 1);
      gpio_put(18, 1);
      gpio_put(17, 1);
      gpio_put(16, 1);
      sleep_ms(5);
    }
  }
  else if (value == "six")
  {
    for (int i = 0; i < 100; ++i)
    {
      gpio_put(20, 0);
      gpio_put(21, 1);
      gpio_put(19, 0);
      gpio_put(18, 1);
      gpio_put(17, 1);
      gpio_put(16, 1);
      sleep_ms(5);
      gpio_put(20, 1);
      gpio_put(21, 0);
      gpio_put(19, 1);
      gpio_put(18, 1);
      gpio_put(17, 1);
      gpio_put(16, 1);
      sleep_ms(5);
    }
  }
  else if (value == "seven")
  {
    for (int i = 0; i < 100; ++i)
    {
      gpio_put(20, 0);
      gpio_put(21, 1);
      gpio_put(19, 1);
      gpio_put(18, 1);
      gpio_put(17, 0);
      gpio_put(16, 0);
      sleep_ms(5);
      gpio_put(20, 1);
      gpio_put(21, 0);
      gpio_put(19, 1);
      gpio_put(18, 0);
      gpio_put(17, 0);
      gpio_put(16, 1);
      sleep_ms(5);
    }
  }
  else if (value == "eight")
  {
    gpio_put(20, 0);
    gpio_put(21, 0);
    gpio_put(19, 1);
    gpio_put(18, 1);
    gpio_put(17, 1);
    gpio_put(16, 1);
    sleep_ms(1000);
  }
  else if (value == "nine")
  {
    for (int i = 0; i < 100; ++i)
    {
      gpio_put(20, 0);
      gpio_put(21, 1);
      gpio_put(19, 1);
      gpio_put(18, 1);
      gpio_put(17, 0);
      gpio_put(16, 0);
      sleep_ms(5);
      gpio_put(20, 1);
      gpio_put(21, 0);
      gpio_put(19, 1);
      gpio_put(18, 1);
      gpio_put(17, 1);
      gpio_put(16, 1);
      sleep_ms(5);
    }
  }
}

// Toggles the built-in LED every inference, and lights a colored LED depending
// on which word was detected.
bool RespondToCommand(tflite::ErrorReporter* error_reporter,
                      int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
  static bool is_initialized = false;
  if (!is_initialized) {
    
    is_initialized = true;
  }
  static int32_t last_command_time = 0;
  static int count = 0;
  static int certainty = 220;

  if (score < 100)
  {
    return false;
  }

  if (score > 120 && current_time - last_command_time > 700
      && found_command != "unknown" && found_command != "silence") {
    printf("Successful inference, heard %s (%d) @%dms\n", found_command,
                         score, current_time);
    last_command_time = current_time;
    // If we hear a command, light up the LEDs
    RunLEDs(found_command);

  }

  // If last_command_time is non-zero but was >3 seconds ago, zero it
  // and switch off the LED.
  printf("General inference, heard %s (%d) @%dms\n", found_command,
                         score, current_time);

  
  return true;
}

#endif  // ARDUINO_EXCLUDE_CODE
