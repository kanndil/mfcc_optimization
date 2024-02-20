# mfcc_optimization

## Description
Optimizing (Mel-Frequency Cepstral Coefficients) algorithm for embedded audio machine learning applications.

## Usage
To compile the project, follow these steps:

1. Clone or download the project repository.
2. Navigate to the kws_nnom_example directory.
3. Open a terminal or command prompt.
4. Run the following command to compile the project:
    ```bash
    make
    ```
5. After successful compilation, an executable named `my_program` will be generated in the project directory.
6. Run the executable:
    ```bash
    ./my_program
    ```

## Structure
The kws_nnom_example directory structure is as follows:

- `mfcc_src/`: Source files for the MFCC module.
- `nnom_src/`: Source files for the NNoM (Neural Networks on Microcontrollers) module.
- `mfcc_inc/`: Header files for the MFCC module.
- `nnom_inc/`: Header files for the NNoM module.
- `test_audio/`: The test dataset for the kws model.
- `main.c`: Main source file of the program.

## Dependencies
* This project requires the GNU Make tool and a C compiler (e.g., gcc) to be installed on your system.
* Follow the README instructions in the `test_audio/` directory to download the test data set

## Cleaning Up
To clean up the compiled files and the executable, run the following command:

    make clean


<!--# Authors

* [Mohamed Shalan](https://github.com/shalan)
* [Youssef Kandil](https://github.com/kanndil)
<br>
<br>-->

## ⚖️ Copyright and Licensing

Copyright 2024 AUC Open Source Hardware Lab

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.