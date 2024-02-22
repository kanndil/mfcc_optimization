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

/* fix_fft.c - Fixed-point in-place Fast Fourier Transform  */
/*
  All data are fixed-point short integers, in which -32768
  to +32768 represent -1.0 to +1.0 respectively. Integer
  arithmetic is used for speed, instead of the more natural
  floating-point.

  For the forward FFT (time -> freq), fixed scaling is
  performed to prevent arithmetic overflow, and to map a 0dB
  sine/cosine wave (i.e. amplitude = 32767) to two -6dB freq
  coefficients. The return value is always 0.

  For the inverse FFT (freq -> time), fixed scaling cannot be
  done, as two 0dB coefficients would sum to a peak amplitude
  of 64K, overflowing the 32k range of the fixed-point integers.
  Thus, the fix_fft() routine performs variable scaling, and
  returns a value which is the number of bits LEFT by which
  the output must be shifted to get the actual amplitude
  (i.e. if fix_fft() returns 3, each value of fr[] and fi[]
  must be multiplied by 8 (2**3) for proper scaling.
  Clearly, this cannot be done within fixed-point short
  integers. In practice, if the result is to be used as a
  filter, the scale_shift can usually be ignored, as the
  result will be approximately correctly normalized as is.

  Written by:  Tom Roberts  11/8/89
  Made portable:  Malcolm Slaney 12/15/94 malcolm@interval.com
  Enhanced:  Dimitrios P. Bouras  14 Jun 2006 dbouras@ieee.org
*/
#define FFT_N 512
#define N_WAVE      1024    /* full length of Sinewave[] */
#define LOG2_N_WAVE 10      /* log2(N_WAVE) */
/*
  Henceforth "short" implies 16-bit word. If this is not
  the case in your architecture, please replace "short"
  with a type definition which *is* a 16-bit word.
*/

/*
  Since we only use 3/4 of N_WAVE, we define only
  this many samples, in order to conserve data space.
*/
short Sinewave[N_WAVE-N_WAVE/4] = {
      0,    201,    402,    603,    804,   1005,   1206,   1406,
   1607,   1808,   2009,   2209,   2410,   2610,   2811,   3011,
   3211,   3411,   3611,   3811,   4011,   4210,   4409,   4608,
   4807,   5006,   5205,   5403,   5601,   5799,   5997,   6195,
   6392,   6589,   6786,   6982,   7179,   7375,   7571,   7766,
   7961,   8156,   8351,   8545,   8739,   8932,   9126,   9319,
   9511,   9703,   9895,  10087,  10278,  10469,  10659,  10849,
  11038,  11227,  11416,  11604,  11792,  11980,  12166,  12353,
  12539,  12724,  12909,  13094,  13278,  13462,  13645,  13827,
  14009,  14191,  14372,  14552,  14732,  14911,  15090,  15268,
  15446,  15623,  15799,  15975,  16150,  16325,  16499,  16672,
  16845,  17017,  17189,  17360,  17530,  17699,  17868,  18036,
  18204,  18371,  18537,  18702,  18867,  19031,  19194,  19357,
  19519,  19680,  19840,  20000,  20159,  20317,  20474,  20631,
  20787,  20942,  21096,  21249,  21402,  21554,  21705,  21855,
  22004,  22153,  22301,  22448,  22594,  22739,  22883,  23027,
  23169,  23311,  23452,  23592,  23731,  23869,  24006,  24143,
  24278,  24413,  24546,  24679,  24811,  24942,  25072,  25201,
  25329,  25456,  25582,  25707,  25831,  25954,  26077,  26198,
  26318,  26437,  26556,  26673,  26789,  26905,  27019,  27132,
  27244,  27355,  27466,  27575,  27683,  27790,  27896,  28001,
  28105,  28208,  28309,  28410,  28510,  28608,  28706,  28802,
  28897,  28992,  29085,  29177,  29268,  29358,  29446,  29534,
  29621,  29706,  29790,  29873,  29955,  30036,  30116,  30195,
  30272,  30349,  30424,  30498,  30571,  30643,  30713,  30783,
  30851,  30918,  30984,  31049,  31113,  31175,  31236,  31297,
  31356,  31413,  31470,  31525,  31580,  31633,  31684,  31735,
  31785,  31833,  31880,  31926,  31970,  32014,  32056,  32097,
  32137,  32176,  32213,  32249,  32284,  32318,  32350,  32382,
  32412,  32441,  32468,  32495,  32520,  32544,  32567,  32588,
  32609,  32628,  32646,  32662,  32678,  32692,  32705,  32717,
  32727,  32736,  32744,  32751,  32757,  32761,  32764,  32766,
  32767,  32766,  32764,  32761,  32757,  32751,  32744,  32736,
  32727,  32717,  32705,  32692,  32678,  32662,  32646,  32628,
  32609,  32588,  32567,  32544,  32520,  32495,  32468,  32441,
  32412,  32382,  32350,  32318,  32284,  32249,  32213,  32176,
  32137,  32097,  32056,  32014,  31970,  31926,  31880,  31833,
  31785,  31735,  31684,  31633,  31580,  31525,  31470,  31413,
  31356,  31297,  31236,  31175,  31113,  31049,  30984,  30918,
  30851,  30783,  30713,  30643,  30571,  30498,  30424,  30349,
  30272,  30195,  30116,  30036,  29955,  29873,  29790,  29706,
  29621,  29534,  29446,  29358,  29268,  29177,  29085,  28992,
  28897,  28802,  28706,  28608,  28510,  28410,  28309,  28208,
  28105,  28001,  27896,  27790,  27683,  27575,  27466,  27355,
  27244,  27132,  27019,  26905,  26789,  26673,  26556,  26437,
  26318,  26198,  26077,  25954,  25831,  25707,  25582,  25456,
  25329,  25201,  25072,  24942,  24811,  24679,  24546,  24413,
  24278,  24143,  24006,  23869,  23731,  23592,  23452,  23311,
  23169,  23027,  22883,  22739,  22594,  22448,  22301,  22153,
  22004,  21855,  21705,  21554,  21402,  21249,  21096,  20942,
  20787,  20631,  20474,  20317,  20159,  20000,  19840,  19680,
  19519,  19357,  19194,  19031,  18867,  18702,  18537,  18371,
  18204,  18036,  17868,  17699,  17530,  17360,  17189,  17017,
  16845,  16672,  16499,  16325,  16150,  15975,  15799,  15623,
  15446,  15268,  15090,  14911,  14732,  14552,  14372,  14191,
  14009,  13827,  13645,  13462,  13278,  13094,  12909,  12724,
  12539,  12353,  12166,  11980,  11792,  11604,  11416,  11227,
  11038,  10849,  10659,  10469,  10278,  10087,   9895,   9703,
   9511,   9319,   9126,   8932,   8739,   8545,   8351,   8156,
   7961,   7766,   7571,   7375,   7179,   6982,   6786,   6589,
   6392,   6195,   5997,   5799,   5601,   5403,   5205,   5006,
   4807,   4608,   4409,   4210,   4011,   3811,   3611,   3411,
   3211,   3011,   2811,   2610,   2410,   2209,   2009,   1808,
   1607,   1406,   1206,   1005,    804,    603,    402,    201,
      0,   -201,   -402,   -603,   -804,  -1005,  -1206,  -1406,
  -1607,  -1808,  -2009,  -2209,  -2410,  -2610,  -2811,  -3011,
  -3211,  -3411,  -3611,  -3811,  -4011,  -4210,  -4409,  -4608,
  -4807,  -5006,  -5205,  -5403,  -5601,  -5799,  -5997,  -6195,
  -6392,  -6589,  -6786,  -6982,  -7179,  -7375,  -7571,  -7766,
  -7961,  -8156,  -8351,  -8545,  -8739,  -8932,  -9126,  -9319,
  -9511,  -9703,  -9895, -10087, -10278, -10469, -10659, -10849,
 -11038, -11227, -11416, -11604, -11792, -11980, -12166, -12353,
 -12539, -12724, -12909, -13094, -13278, -13462, -13645, -13827,
 -14009, -14191, -14372, -14552, -14732, -14911, -15090, -15268,
 -15446, -15623, -15799, -15975, -16150, -16325, -16499, -16672,
 -16845, -17017, -17189, -17360, -17530, -17699, -17868, -18036,
 -18204, -18371, -18537, -18702, -18867, -19031, -19194, -19357,
 -19519, -19680, -19840, -20000, -20159, -20317, -20474, -20631,
 -20787, -20942, -21096, -21249, -21402, -21554, -21705, -21855,
 -22004, -22153, -22301, -22448, -22594, -22739, -22883, -23027,
 -23169, -23311, -23452, -23592, -23731, -23869, -24006, -24143,
 -24278, -24413, -24546, -24679, -24811, -24942, -25072, -25201,
 -25329, -25456, -25582, -25707, -25831, -25954, -26077, -26198,
 -26318, -26437, -26556, -26673, -26789, -26905, -27019, -27132,
 -27244, -27355, -27466, -27575, -27683, -27790, -27896, -28001,
 -28105, -28208, -28309, -28410, -28510, -28608, -28706, -28802,
 -28897, -28992, -29085, -29177, -29268, -29358, -29446, -29534,
 -29621, -29706, -29790, -29873, -29955, -30036, -30116, -30195,
 -30272, -30349, -30424, -30498, -30571, -30643, -30713, -30783,
 -30851, -30918, -30984, -31049, -31113, -31175, -31236, -31297,
 -31356, -31413, -31470, -31525, -31580, -31633, -31684, -31735,
 -31785, -31833, -31880, -31926, -31970, -32014, -32056, -32097,
 -32137, -32176, -32213, -32249, -32284, -32318, -32350, -32382,
 -32412, -32441, -32468, -32495, -32520, -32544, -32567, -32588,
 -32609, -32628, -32646, -32662, -32678, -32692, -32705, -32717,
 -32727, -32736, -32744, -32751, -32757, -32761, -32764, -32766,
};

/*
  FIX_MPY() - fixed-point multiplication & scaling.
  Substitute inline assembly for hardware-specific
  optimization suited to a particluar DSP processor.
  Scaling ensures that result remains 16-bit.
*/
 short FIX_MPY(short a, short b)
{
    /* shift right one less bit (i.e. 15-1) */
    int c = ((int)a * (int)b) >> 14;
    /* last bit shifted out = rounding-bit */
    b = c & 0x01;
    /* last shift + rounding bit */
    a = (c >> 1) + b;
    return a;
}

/*
  fix_fft() - perform forward/inverse fast Fourier transform.
  fr[n],fi[n] are real and imaginary arrays, both INPUT AND
  RESULT (in-place FFT), with 0 <= n < 2**m; set inverse to
  0 for forward transform (FFT), or 1 for iFFT.
*/
int fix_fft(short fr[], short fi[], short m, short inverse)
{
    int mr, nn, i, j, l, k, istep, n, scale, shift;
    short qr, qi, tr, ti, wr, wi;

    n = 1 << m;

    /* max FFT size = N_WAVE */
    if (n > N_WAVE)
        return -1;

    mr = 0;
    nn = n - 1;
    scale = 0;

    /* decimation in time - re-order data */
    for (m=1; m<=nn; ++m) {
        l = n;
        do {
            l >>= 1;
        } while (mr+l > nn);
        mr = (mr & (l-1)) + l;

        if (mr <= m)
            continue;
        tr = fr[m];
        fr[m] = fr[mr];
        fr[mr] = tr;
        ti = fi[m];
        fi[m] = fi[mr];
        fi[mr] = ti;
    }

    l = 1;
    k = LOG2_N_WAVE-1;
    while (l < n) {
        if (inverse) {
            /* variable scaling, depending upon data */
            shift = 0;
            for (i=0; i<n; ++i) {
                j = fr[i];
                if (j < 0)
                    j = -j;
                m = fi[i];
                if (m < 0)
                    m = -m;
                if (j > 16383 || m > 16383) {
                    shift = 1;
                    break;
                }
            }
            if (shift)
                ++scale;
        } else {
            /*
              fixed scaling, for proper normalization --
              there will be log2(n) passes, so this results
              in an overall factor of 1/n, distributed to
              maximize arithmetic accuracy.
            */
            shift = 1;
        }
        /*
          it may not be obvious, but the shift will be
          performed on each data point exactly once,
          during this pass.
        */
        istep = l << 1;
        for (m=0; m<l; ++m) {
            j = m << k;
            /* 0 <= j < N_WAVE/2 */
            wr =  Sinewave[j+N_WAVE/4];
            wi = -Sinewave[j];
            if (inverse)
                wi = -wi;
            if (shift) {
                wr >>= 1;
                wi >>= 1;
            }
            for (i=m; i<n; i+=istep) {
                j = i + l;
                tr = FIX_MPY(wr,fr[j]) - FIX_MPY(wi,fi[j]);
                ti = FIX_MPY(wr,fi[j]) + FIX_MPY(wi,fr[j]);
                qr = fr[i];
                qi = fi[i];
                if (shift) {
                    qr >>= 1;
                    qi >>= 1;
                }
                fr[j] = qr - tr;
                fi[j] = qi - ti;
                fr[i] = qr + tr;
                fi[i] = qi + ti;
            }
        }
        --k;
        l = istep;
    }
    return scale;
}

/*
  fix_fftr() - forward/inverse FFT on array of real numbers.
  Real FFT/iFFT using half-size complex FFT by distributing
  even/odd samples into real/imaginary arrays respectively.
  In order to save data space (i.e. to avoid two arrays, one
  for real, one for imaginary samples), we proceed in the
  following two steps: a) samples are rearranged in the real
  array so that all even samples are in places 0-(N/2-1) and
  all imaginary samples in places (N/2)-(N-1), and b) fix_fft
  is called with fr and fi pointing to index 0 and index N/2
  respectively in the original array. The above guarantees
  that fix_fft "sees" consecutive real samples as alternating
  real and imaginary samples in the complex array.
*/
int fix_fftr(short f[], int m, int inverse)
{
    int i, N = 1<<(m-1), scale = 0;
    short tt, *fr=f, *fi=&f[N];

    if (inverse)
        scale = fix_fft(fi, fr, m-1, inverse);
    for (i=1; i<N; i+=2) {
        tt = f[N+i-1];
        f[N+i-1] = f[i];
        f[i] = tt;
    }
    if (! inverse)
        scale = fix_fft(fi, fr, m-1, inverse);
    return scale;
}



// Function to convert floating-point data to fixed-point
void float_to_fixed(float float_data[], short fixed_data[], int size) {
    for (int i = 0; i < size; ++i) {
        // Scale and convert to fixed-point representation
        fixed_data[i] = (short)(float_data[i] );
        fixed_data[i] = (short)(float_data[i] * 32767.0f);
    }
}

// Function to convert fixed-point data to floating-point
void fixed_to_float(short fixed_data[], float float_data[], int size) {
    for (int i = 0; i < size; ++i) {
        // Convert back to floating-point representation
        float_data[i] = (float)fixed_data[i] / 32767.0f;
    }
}




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
    mfcc->frame[0] = last;

    for (i = 1; i < mfcc->frame_len; i++) {
        mfcc->frame[i] = ((float)audio_data[i] - last * mfcc->preempha);
        last = (float)audio_data[i];
    }
    //Fill up remaining with zeros
    if(mfcc->frame_len_padded - mfcc->frame_len) //todo: replace memset with loop
        memset(&mfcc->frame[mfcc->frame_len], 0, sizeof(float) * (mfcc->frame_len_padded - mfcc->frame_len));


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
    short data_re2[512];
    short data_im2[512];
 
    for (int i = 0; i < 512; ++i) {
        data_re2[i] = (short)data_re[i]; // Casting each float to short
        data_im2[i] = (short)data_im[i]; // Casting each float to short
    }
    
    fix_fft(data_re2, data_im2, log2(FFT_N), 0);
//    fixed_to_float(data_re2, data_re, 512);
//    fixed_to_float(data_im2, data_im, 512);
    
    
   // FFT data structure
    // only need half (N/2+1)
    for (int i = 0; i <= mfcc->frame_len_padded/2; i++) {
        mfcc->buffer[i] = (data_re2[i] * data_re2[i] + data_im2[i]* data_im2[i]);
    }
    for (int i = 0; i <= mfcc->frame_len_padded/2; i++) {
        mfcc->buffer[i] /= 32767.0f;
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
    float tempout[13]={0};
    for (i = mfcc->num_features_offset; i < mfcc->num_mfcc_features; i++)
    {
        float sum = 0.0;
        for (j = 0; j < mfcc->num_fbank ; j++)
        {
            sum += mfcc->dct_matrix[i*mfcc->num_fbank +j] * mfcc->mel_energies[j];
        }
        mfcc_out[out_index] = sum;
        tempout[out_index] = sum;
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
