/*
 * Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modified by Jianjia Ma for C implementation.
 *
 */

/*
 * Description: MFCC feature extraction to match with TensorFlow MFCC Op
 */

#include <string.h>
#include <stdlib.h>

#include "mfcc.h"
#include "float.h"


#define M_PI 3.14159265358979323846264338327950288

#ifndef MFCC_PLATFORM_ARM
// FFT code from arduino_fft: https://github.com/lloydroc/arduino_fft
// change to float data£¬ modify to fit within this file
// see the above link for license( MIT license).
#include <stdio.h>
#include <math.h>




// hz --> mel
static inline float MelScale(float freq) {
    // TODO: remove the the 2595 foctor to reduce operations, and to ovoid int overflow
  return 2595.0f * log10(1.0f + freq / 700.0f);
}

// mel --> hz
static inline float InverseMelScale(float mel_freq) {
    // TODO: remove the the 2595 foctor to reduce operations, and to ovoid int overflow
  return 700.0f * (pow(10,(mel_freq / 2595.0f)) - 1.0f);
}

void rearrange(float data_re[], float data_im[], const unsigned int N)
{
    unsigned int target = 0;
    for (unsigned int position = 0; position < N; position++)
    {
        if (target > position) {
            const float temp_re = data_re[target];
            const float temp_im = data_im[target];
            data_re[target] = data_re[position];
            data_im[target] = data_im[position];
            data_re[position] = temp_re;
            data_im[position] = temp_im;
        }
        unsigned int mask = N;
        while (target & (mask >>= 1))
            target &= ~mask;
        target |= mask;
    }
}

void compute(float data_re[], float data_im[], const unsigned int N)
{
    const float pi = -3.14159265358979323846;
    for (unsigned int step = 1; step < N; step <<= 1) {
        const unsigned int jump = step << 1;
        const float step_d = (float)step;
        float twiddle_re = 1.0;
        float twiddle_im = 0.0;
        for (unsigned int group = 0; group < step; group++)
        {
            for (unsigned int pair = group; pair < N; pair += jump)
            {
                const unsigned int match = pair + step;
                const float product_re = twiddle_re * data_re[match] - twiddle_im * data_im[match];
                const float product_im = twiddle_im * data_re[match] + twiddle_re * data_im[match];
                data_re[match] = data_re[pair] - product_re;
                data_im[match] = data_im[pair] - product_im;
                data_re[pair] += product_re;
                data_im[pair] += product_im;
            }
            // we need the factors below for the next iteration
            // if we don't iterate then don't compute
            if (group + 1 == step)
            {
                continue;
            }
            float angle = pi * ((float)group + 1) / step_d;
            twiddle_re = cosf(angle);
            twiddle_im = sinf(angle);
        }
    }
}

void fft(float data_re[], float data_im[], const int N)
{
    rearrange(data_re, data_im, N);
    compute(data_re, data_im, N);
}

#endif /* end of FFT implmentation*/

void mfcc_create(mfcc_t *mfcc,int num_mfcc_features, int feature_offset, int num_fbank, int frame_len, float preempha, int is_append_energy)
{

    /*  This is the methodology of processing the MFCC  */
    
    /*
       **************************************************
       *                    Waveform                    *
       *                       |                        *
       *                       |                        *
       *                       V                        *
       *                   DFT OR FFT                   *
       *                       |                        *
       *                       |                        *
       *                       V                        *
       *             Log-Amplitude Spectrum             *
       *                       |                        *
       *                       |                        *
       *                       V                        *
       *                  Mel-Scaling                   *
       *                       |                        *
       *                       |                        *
       *                       V                        *
       *         Discrete Cosine Transform (DCT)        *
       *                       |                        *
       *                       |                        *
       *                       V                        *
       *                     MFCCs                      *
       **************************************************
     */
    
    
    
    mfcc->num_mfcc_features = num_mfcc_features;
    mfcc->num_features_offset = feature_offset;
    mfcc->num_fbank = num_fbank;
    mfcc->frame_len = frame_len;
    mfcc->preempha = preempha;
    mfcc->is_append_energy = is_append_energy;

    // Round-up to nearest power of 2.
    mfcc->frame_len_padded = 512;

    //create window function, hanning
    // By processing data through HANNING before applying FFT, more realistic results can be obtained.
    for (int i = 0; i < frame_len; i++)
        mfcc->window_func[i] = 0.5f - 0.5f*cosf((float)M_2PI * ((float)i) / (frame_len));

    //create mel filterbank
    create_mel_fbank(mfcc);

    //create DCT matrix
    create_dct_matrix(mfcc->num_fbank , num_mfcc_features, mfcc);

#ifdef MFCC_PLATFORM_ARM
    
    // MARK: - this can be static // depends on the Hardware
    //initialize FFT
    mfcc->rfft = mfcc_malloc(sizeof(arm_rfft_fast_instance_f32));
    arm_rfft_fast_init_f32(mfcc->rfft, mfcc->frame_len_padded);
#else

#endif
    return;
}


void create_dct_matrix(int32_t input_length, int32_t coefficient_count, mfcc_t* mfcc)
{
    int32_t k, n;
    float normalizer;
#ifdef MFCC_PLATFORM_ARM
    arm_sqrt_f32(2.0f/(float)input_length, &normalizer);
#else
    normalizer = sqrtf(2.0f/(float)input_length);
#endif
    for (k = 0; k < coefficient_count; k++)
    {
        for (n = 0; n < input_length; n++)
        {
            mfcc->dct_matrix[k*input_length+n] = normalizer * cosf( ((float)M_PI)/input_length * (n + 0.5f) * k );
        }
    }
    return;
}

void create_mel_fbank(mfcc_t *mfcc) {

    // compute points evenly spaced in mels
    float mel_low_freq = MelScale(MEL_LOW_FREQ);                                    // MARK: - this can be fixed
    float mel_high_freq = MelScale(MEL_HIGH_FREQ);                                  // MARK: - this can be fixed
    float mel_freq_delta = (mel_high_freq - mel_low_freq) / (mfcc->num_fbank +1);   // MARK: - this can be fixed

//    float bin[28];
    for (int i=0; i<mfcc->num_fbank+2; i++)
    {
        mfcc->mel_fbins[i] = mel_low_freq + mel_freq_delta*i;
        mfcc->mel_fbins[i] = floor((mfcc->frame_len_padded+1)*InverseMelScale(mfcc->mel_fbins[i])/SAMP_FREQ);
    }

    return;

}
void mfcc_compute(mfcc_t *mfcc, const int16_t * audio_data, float* mfcc_out)
{
    int32_t i, j, bin;

    //1. TensorFlow way of normalizing .wav data to (-1,1) and 2. do pre-emphasis.
    float last = (float)audio_data[0];
    mfcc->frame[0] = last / (1 << 15);  
    //todo: remove the normalization step, we need the input in fixed format
    //todo: since the input is already in fixed format, we can remove the normalization step

    for (i = 1; i < mfcc->frame_len; i++) {
        mfcc->frame[i] = ((float)audio_data[i] - last * mfcc->preempha) / (1<<15); 
        last = (float)audio_data[i];
        //todo: remove the normalization step
        //todo: leave the empha step
        //todo: check that the preempha is in fixed format
    }
    //Fill up remaining with zeros
    if(mfcc->frame_len_padded - mfcc->frame_len) //todo: replace memset with loop
        memset(&mfcc->frame[mfcc->frame_len], 0, sizeof(float) * (mfcc->frame_len_padded - mfcc->frame_len));

//     windows filter
//    for (i = 0; i < mfcc->frame_len; i++) {
//        mfcc->frame[i] *= mfcc->window_func[i];
//    }

#ifdef MFCC_PLATFORM_ARM // ToDo add other fft implementation
    //Compute FFT
    arm_rfft_fast_f32(mfcc->rfft, mfcc->frame, mfcc->buffer, 0);

    //Convert to power spectrum
    //frame is stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
    int32_t half_dim = mfcc->frame_len_padded/2;
    float first_energy = mfcc->buffer[0] * mfcc->buffer[0];
    float last_energy = mfcc->buffer[1] * mfcc->buffer[1];  // handle this special case
    for (i = 1; i < half_dim; i++) {
        float real = mfcc->buffer[i*2];
        float im = mfcc->buffer[i*2 + 1];
        mfcc->buffer[i] = real*real + im*im;
    }
    mfcc->buffer[0] = first_energy;
    mfcc->buffer[half_dim] = last_energy;

#else // end of ARM_fft
    // not yet optimized for memory
    float *data_re = mfcc->frame;
    float data_im[512] = {0};

    fft(data_re, data_im, mfcc->frame_len_padded);
    
    
   // FFT data structure
    // only need half (N/2+1)
    for (int i = 0; i <= mfcc->frame_len_padded/2; i++) {
        mfcc->buffer[i] = (data_re[i] * data_re[i] + data_im[i]* data_im[i])/mfcc->frame_len_padded;
    }
#endif

    //Apply mel filterbanks
    apply_filter_banks(mfcc);
    

    //Take log
    float total_energy = 0;
    for (bin = 0; bin < mfcc->num_fbank; bin++)
    {
        total_energy += mfcc->mel_energies[bin];
        mfcc->mel_energies[bin] = logf(mfcc->mel_energies[bin]);
    }
    //Take DCT. Uses matrix mul.
    int out_index = 0;
    for (i = mfcc->num_features_offset; i < mfcc->num_mfcc_features; i++)
    {
        float sum = 0.0;
        for (j = 0; j < mfcc->num_fbank ; j++)
        {
            sum += mfcc->dct_matrix[i*mfcc->num_fbank +j] * mfcc->mel_energies[j];
        }
        mfcc_out[out_index] = sum;
        out_index ++;
    }

}


void apply_filter_banks(mfcc_t *mfcc){

    for (int j = 0; j < mfcc->num_fbank ; j++){
        float left = mfcc->mel_fbins[j];
        float center = mfcc->mel_fbins[j + 1];
        float right = mfcc->mel_fbins[j + 2];
        float mel_energy = 0;

        for (int i = left + 1; i < center; i++) {
            mel_energy += mfcc->buffer[i] * (i - left) / (center - left);
        }
        for (int i = center; i < right; i++) {
            mel_energy += mfcc->buffer[i] * (right - i) / (right - center);
        }
        if (mel_energy == 0.0f)
            mel_energy = FLT_MIN;
        mfcc->mel_energies[j] = mel_energy;
    }
    


}
