/*
 * Copyright (c) 2018-2020, Jianjia Ma
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-03-29     Jianjia Ma   first implementation
 *
 * Notes:
 * This is a keyword spotting example using NNoM
 *
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "nnom.h"
#include "kws_weights.h"

#include "mfcc.h"
#include "math.h"


// NNoM model
nnom_model_t *model;

// 10 labels-1
//const char label_name[][10] =  {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknow"};

// 10 labels-2
//const char label_name[][10] =  {"marvin", "sheila", "yes", "no", "left", "right", "forward", "backward", "stop", "go", "unknow"};

// full 34 labels
const char label_name[][10] =  {"backward", "bed", "bird", "cat", "dog", "down", "eight","five", "follow", "forward",
                      "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right",
                      "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "yes", "zero", "unknow"};



mfcc_t mfcc;
char ground_truth[12000][10];
#define AUDIO_FRAME_LEN (512) //31.25ms * 16000hz = 512, // FFT (windows size must be 2 power n)
int dma_audio_buffer[AUDIO_FRAME_LEN]; //512
int16_t audio_buffer_16bit[(int)(AUDIO_FRAME_LEN*1.5)]; // an easy method for 50% overlapping
int audio_sample_i = 0;

//the mfcc feature for kws
#define MFCC_LEN            (62)
#define MFCC_COEFFS_FIRST   (1)     // ignore the mfcc feature before this number
#define MFCC_COEFFS_LEN     (13)    // the total coefficient to calculate
#define MFCC_TOTAL_NUM_BANK (26)    // total number of filter bands
#define MFCC_COEFFS         (MFCC_COEFFS_LEN-MFCC_COEFFS_FIRST)

#define MFCC_FEAT_SIZE  (MFCC_LEN * MFCC_COEFFS)
float mfcc_features_f[MFCC_COEFFS];             // output of mfcc
int8_t mfcc_features[MFCC_LEN][MFCC_COEFFS];     // ring buffer
int8_t mfcc_features_seq[MFCC_LEN][MFCC_COEFFS]; // sequencial buffer for neural network input.
uint32_t mfcc_feat_index = 0;
#define MAX_SIZE 40001 // Assuming integers are in the range -1000 to 1000

void quantize_data(float*din, int8_t *dout, uint32_t size, uint32_t int_bit);
void thread_kws_serv(void);



int main(void)
{
    // create and compile the model
    model = nnom_model_create();
    
    uint32_t label;
    float prob;
    audio_sample_i = 0;
    int audios_processed = 0;
    int correct_audios = 0;
    FILE * file;
    FILE * ground_truth_f;
    int frame_size = 512;
    
    // Open files for reading audio data and ground truth labels
    file = fopen ("/Users/youssef/Documents/Work/AUC_Open_Hardware_Lab/mfcc_testing/Test_Audio/test_x.txt","r"); //the audio data stored in a textfile
    ground_truth_f = fopen ("/Users/youssef/Documents/Work/AUC_Open_Hardware_Lab/mfcc_testing/Test_Audio/test_y.txt","r"); //the ground truth textfile

    
    // Read ground truth labels from file into an array
    for (int j = 0; fscanf(ground_truth_f, "%s", ground_truth[j++]) != EOF;) ;
    fclose (ground_truth_f);

    // calculate 13 coefficient, use number #2~13 coefficient. discard #1
    // features, offset, bands, 512fft, 0 preempha, attached_energy_to_band0
    mfcc_create(&mfcc, MFCC_COEFFS_LEN, MFCC_COEFFS_FIRST, MFCC_TOTAL_NUM_BANK, AUDIO_FRAME_LEN, 0.97f, true);

    
    // Main loop for processing audio samples and performing inference
    while (1)
    {
        // Read audio samples from file
        for (int p = 0; p < frame_size ; fscanf(file, "%d", &dma_audio_buffer[p++]));
   
        // Process audio samples and compute MFCC features
        thread_kws_serv();
        
        // Update audio sample index and adjust frame size for overlap
        audio_sample_i += frame_size;
        if(audio_sample_i == 15872) //31*512
            frame_size = 128; //0.25*512 // Adjust frame size for 50% overlap

        // Perform inference when enough samples are available
        if(audio_sample_i==16000)
        {
            // Copy MFCC features to input data buffer for the neural network
            memcpy(nnom_input_data, mfcc_features, MFCC_FEAT_SIZE);
            
            // Make a prediction using the neural network
            nnom_predict(model, &label, &prob);
            printf("%d %s : %d%% - Ground Truth is: %s\n", audios_processed, (char*)&label_name[label], (int)(prob * 100),ground_truth[audios_processed]);
            
            // Check if the predicted label matches the ground truth label
            if(strcmp(ground_truth[audios_processed], label_name[label]) == 0)
                correct_audios++;
            
            // Reset audio sample index and frame size
            audio_sample_i = 0;
            audios_processed++;// Increment sample counter
            frame_size = 512;
        }

        // Exit the loop when a sufficient number of samples are processed
        if(audios_processed>=11000) break;
    }
    
    // Calculate and print the overall accuracy
    printf("Accuracy : %.6f%%\n",((float)correct_audios/(audios_processed) * 100));
    
    // Close the file containing audio samples
    fclose(file);
}


void thread_kws_serv(void)
{

        if (audio_sample_i == 15872)
            memset(&dma_audio_buffer[128], 0, sizeof(int) * 384); //to fill the latest quarter in the latest frame

        // memory move
        // audio buffer = | 256 byte old data |   256 byte new data 1 | 256 byte new data 2 |
        //                         ^------------------------------------------|
        memcpy(audio_buffer_16bit, &audio_buffer_16bit[AUDIO_FRAME_LEN], (AUDIO_FRAME_LEN/2)*sizeof(int16_t));

        // convert it to 16 bit.
        // volume*4
        for(int i = 0; i < AUDIO_FRAME_LEN; i++)
        {
            audio_buffer_16bit[AUDIO_FRAME_LEN/2+i] = dma_audio_buffer[i];
        }

        // MFCC
        // do the first mfcc with half old data(256) and half new data(256)
        // then do the second mfcc with all new data(512).
        // take mfcc buffer

        for(int i=0; i<2; i++)
        {
            if ((audio_sample_i != 0 || i==1) && (audio_sample_i != 15872 || i==0)) //to skip computing first mfcc block that's half empty
            {
                mfcc_compute(&mfcc, &audio_buffer_16bit[i*AUDIO_FRAME_LEN/2], mfcc_features_f);


                // quantise them using the same scale as training data (in keras), by 2^n.
                quantize_data(mfcc_features_f, mfcc_features[mfcc_feat_index], MFCC_COEFFS, 3);

                // debug only, to print mfcc data on console
                if(0)
                {
                    for(int q=0; q<MFCC_COEFFS; q++)
                        printf("%d ",  mfcc_features[mfcc_feat_index][q]);
                    printf("\n");
                }
                mfcc_feat_index++;
                if(mfcc_feat_index >= MFCC_LEN)
                    mfcc_feat_index = 0;
            }

        }
}

void quantize_data(float*din, int8_t *dout, uint32_t size, uint32_t int_bit)
{
    #define _MAX(x, y) (((x) > (y)) ? (x) : (y))
    #define _MIN(x, y) (((x) < (y)) ? (x) : (y))
    float limit = (1 << int_bit);
    float d;
    for(uint32_t i=0; i<size; i++)
    {
        d = round(_MAX(_MIN(din[i], limit), -limit) / limit * 128);
        d = d/128.0f;
        dout[i] = round(d *127);
    }
}
