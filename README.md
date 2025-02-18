# mlcard
Speech recognition embedded into an RP2040 on a business card

This uses Pete Warden's fork of [TFLite Micro](https://github.com/usefulsensors/pico-tflmicro) and what looks to be ARM's [PDM microphone library](https://github.com/ArmDeveloperEcosystem/microphone-library-for-pico), and combines them with the [TFLite Micro Arduino Speech example](https://github.com/tensorflow/tflite-micro-arduino-examples/tree/main/examples/micro_speech) to run speech recognition in real time on an RP2040 chip. 

Most of this I did not write myself - I've trained the model to recognise 11 words, instead of the basic 2 words, and changed the code to light up LEDs, etc, or cut out other stuff, but largely this is just using existing repos and combining them. 

The circuit is essentially a Raspberry Pi Pico, without all the breakout pins, with LEDs and a PDM microphone connected. 

This repo assumes the PDM and TFLite micro repos reside in the same directory that contains this repo. 
