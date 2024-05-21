import math
import numpy as np

SAMP_FREQ = 16000
MEL_LOW_FREQ = 20
MEL_HIGH_FREQ = 4000
M_2PI = 6.283185307179586476925286766559005
M_PI  = 3.14159265358979323846264338327950288
FFT_N = 512
N_WAVE = 1024
LOG2_N_WAVE = 10

class MFCC_FIX:
    def __init__(self,):
        self.num_mfcc_features = 0
        self.num_features_offset = 0
        self.num_fbank = 0
        self.frame_len = 0
        self.preempha = 0
        self.is_append_energy = 0
        self.frame_len_padded = 512
        self.frame = np.zeros(512)
        self.buffer = np.zeros(257)
        self.mel_energies = np.zeros(26)
        self.window_func = np.zeros(512)
        self.mel_fbins = np.zeros(28)
        self.dct_matrix = np.zeros(338)




# Sine table
Sinewave = np.array([
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
 -32727, -32736, -32744, -32751, -32757, -32761, -32764, -32766
 ])

def fix_mpy(a, b):
    c = (a * b) >> 14
    b = c & 0x01
    a = (c >> 1) + b
    return a

def fix_fft(fr, fi, m, inverse):
    
    
    n = 1 << m

    if n > N_WAVE:
        return -1
    
    mr = 0
    nn = n - 1
    scale = 0

    for m in range(1, nn + 1):
        l = n
        while True:
            l >>= 1
            if mr + l < nn:
                break
        mr = (mr & (l - 1)) + l
        
        if mr <= m:
            continue
        tr = fr[m]
        fr[m] = fr[mr]
        fr[mr] = tr
        ti = fi[m]
        fi[m] = fi[mr]
        fi[mr] = ti

    l = 1
    k = LOG2_N_WAVE - 1
    while l < n:
        if inverse:
            shift = 0
            for i in range(n):
                j = abs(fr[i])
                m = abs(fi[i])
                if j > 16383 or m > 16383:
                    shift = 1
                    break
            if shift:
                scale += 1
        else:
            shift = 1


        istep = l << 1
        for m in range(l):
            j = m << k
            wr = Sinewave[j + N_WAVE // 4]
            wi = -Sinewave[j]
            if inverse:
                wi = -wi
            if shift:
                wr >>= 1
                wi >>= 1
            for i in range(m, nn + 1, istep):
                j = i + l
                tr = fix_mpy(wr, fr[j]) - fix_mpy(wi, fi[j])
                ti = fix_mpy(wr, fi[j]) + fix_mpy(wi, fr[j])
                qr = fr[i]
                qi = fi[i]
                if shift:
                    qr >>= 1
                    qi >>= 1
                fr[j] = qr - tr
                fi[j] = qi - ti
                fr[i] = qr + tr
                fi[i] = qi + ti
        k -= 1
        l = istep
    return scale


def mel_scale(freq):
    return 2595.0 * math.log10(1.0 + freq / 700.0)

def inverse_mel_scale(mel_freq):
    return 700.0 * (10 ** (mel_freq / 2595.0) - 1.0)


def create_dct_matrix(input_length, coefficient_count, mfcc_fix):
    normalizer = math.sqrt(2.0 / input_length)
    for k in range(coefficient_count):
        for n in range(input_length):
            mfcc_fix.dct_matrix[k * input_length + n] = normalizer * math.cos(M_PI/ input_length * (n + 0.5) * k)

def create_mel_fbank(mfcc_fix):
    mel_low_freq = mel_scale(MEL_LOW_FREQ)
    mel_high_freq = mel_scale(MEL_HIGH_FREQ)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (mfcc_fix.num_fbank + 1)
    for i in range(mfcc_fix.num_fbank + 2):
        mfcc_fix.mel_fbins[i] = mel_low_freq + mel_freq_delta * i
        mfcc_fix.mel_fbins[i] = math.floor((mfcc_fix.frame_len_padded + 1) * inverse_mel_scale(mfcc_fix.mel_fbins[i]) / SAMP_FREQ)

def mfcc_create(mfcc_fix, num_mfcc_features, feature_offset, num_fbank, frame_len, preempha, is_append_energy):
    mfcc_fix.num_mfcc_features = num_mfcc_features
    mfcc_fix.num_features_offset = feature_offset
    mfcc_fix.num_fbank = num_fbank
    mfcc_fix.frame_len = frame_len
    mfcc_fix.preempha = preempha
    mfcc_fix.is_append_energy = is_append_energy

    # Round-up to nearest power of 2.
    mfcc_fix.frame_len_padded = 512

    # create window function, hanning
    # By processing data through HANNING before applying FFT, more realistic results can be obtained.
    mfcc_fix.window_func = np.zeros(frame_len)
    for i in range(frame_len):
        mfcc_fix.window_func[i] = 0.5 - 0.5 * np.cos(M_2PI * i / frame_len)

    # create mel filterbank
    create_mel_fbank(mfcc_fix)

    # create DCT matrix
    create_dct_matrix(mfcc_fix.num_fbank, num_mfcc_features, mfcc_fix)

    return

def apply_filter_banks(mfcc_fix):
    for j in range(mfcc_fix.num_fbank):
        left = mfcc_fix.mel_fbins[j]
        center = mfcc_fix.mel_fbins[j + 1]
        right = mfcc_fix.mel_fbins[j + 2]
        mel_energy = 0
        for i in range(int(left) + 1, int(center)):
            mel_energy += mfcc_fix.buffer[i] * (i - left) / (center - left)
        for i in range(int(center), int(right)):
            mel_energy += mfcc_fix.buffer[i] * (right - i) / (right - center)
        if mel_energy == 0.0:
            mel_energy = np.finfo(float).tiny
        mfcc_fix.mel_energies[j] = mel_energy

def mfcc_compute(mfcc_fix, audio_data, mfcc_out):
    
    last = float(audio_data[0])
    mfcc_fix.frame[0] = last
    for i in range(1, mfcc_fix.frame_len):
        mfcc_fix.frame[i] = float(audio_data[i]) - last * mfcc_fix.preempha
        last = float(audio_data[i])
    if mfcc_fix.frame_len_padded - mfcc_fix.frame_len:
        mfcc_fix.frame[mfcc_fix.frame_len:] = 0
        
    for i in range(mfcc_fix.frame_len):
        mfcc_fix.frame[i] *= mfcc_fix.window_func[i]


    data_re = mfcc_fix.frame
    data_im = np.zeros(512)
    data_re2 = data_re.astype(np.int16)
    data_im2 = data_im.astype(np.int16)
    m = int(math.log2(FFT_N))
    fix_fft(data_re2, data_im2, m, 0)

    for i in range(mfcc_fix.frame_len_padded // 2 + 1):
        temp1 = (float(data_re2[i]) * float(data_re2[i]))
        temp2 = (float(data_im2[i]) * float(data_im2[i]))
        mfcc_fix.buffer[i] = temp2 + temp1
    for i in range(mfcc_fix.frame_len_padded // 2 + 1):
        mfcc_fix.buffer[i] /= 32767.0
        
    apply_filter_banks(mfcc_fix)

    total_energy = 0
    for bin in range(mfcc_fix.num_fbank):
        total_energy += mfcc_fix.mel_energies[bin]
        mfcc_fix.mel_energies[bin] = math.log(mfcc_fix.mel_energies[bin])

    out_index = 0
    for i in range(mfcc_fix.num_features_offset, mfcc_fix.num_mfcc_features):
        sum_val = 0.0
        for j in range(mfcc_fix.num_fbank):
            sum_val += mfcc_fix.dct_matrix[i * mfcc_fix.num_fbank + j] * mfcc_fix.mel_energies[j]
        mfcc_out[out_index] = sum_val
        out_index += 1


def abs_mean(p):
    sum_val = 0
    size = len(p)
    for i in range(size):
        if p[i] < 0:
            sum_val += -p[i]
        else:
            sum_val += p[i]
    return sum_val // size

def quantize_data(din, int_bit):
    size = len(din)
    limit = 1 << int_bit
    dout = np.zeros(size, dtype=np.int8)
    for i in range(size):
        d = round(max(min(din[i], limit), -limit) / limit * 128)
        d /= 128.0
        dout[i] = round(d * 127)
    return dout



SAMP_FREQ = 16000
AUDIO_FRAME_LEN = 512

mfcc_fix = MFCC_FIX()

dma_audio_buffer = np.zeros(AUDIO_FRAME_LEN, dtype=np.int32)
audio_buffer_16bit = np.zeros(int(AUDIO_FRAME_LEN * 1.5), dtype=np.int16)
audio_sample_i = 0

MFCC_LEN = 62
MFCC_COEFFS_FIRST = 1
MFCC_COEFFS_LEN = 13
MFCC_COEFFS = MFCC_COEFFS_LEN - MFCC_COEFFS_FIRST
MFCC_FEAT_SIZE = MFCC_LEN * MFCC_COEFFS

mfcc_features_f = np.zeros(MFCC_COEFFS)
mfcc_features = np.zeros((MFCC_LEN, MFCC_COEFFS), dtype=np.int8)
mfcc_feat_index = 0


def thread_kws_serv():
    global audio_sample_i
    global dma_audio_buffer
    global audio_buffer_16bit
    global mfcc_feat_index
    global mfcc_features_f

    if audio_sample_i == 15872:
        dma_audio_buffer[128:512] = 0  # to fill the latest quarter in the latest frame

    audio_buffer_16bit[:AUDIO_FRAME_LEN // 2] = audio_buffer_16bit[AUDIO_FRAME_LEN:]

    audio_buffer_16bit[AUDIO_FRAME_LEN // 2:] = dma_audio_buffer

    for i in range(2):
        if (audio_sample_i != 0 or i == 1) and (audio_sample_i != 15872 or i == 0):
            mfcc_compute(mfcc_fix, audio_buffer_16bit[i * (AUDIO_FRAME_LEN // 2): ], mfcc_features_f)

            # quantize them using the same scale as training data (in keras), by 2^n.
            mfcc_features[mfcc_feat_index] = quantize_data(mfcc_features_f, 3)

            # debug only, to print mfcc_fix data on console
            if 0:
                print("MFCC Features:")
                print(mfcc_features[mfcc_feat_index])

            mfcc_feat_index += 1
            if mfcc_feat_index >= MFCC_LEN:
                mfcc_feat_index = 0


def process_audio_data(data):
    global audio_sample_i
    global dma_audio_buffer
    global audio_buffer_16bit
    global mfcc_feat_index
    global mfcc_features_f
    mfcc_create(mfcc_fix, MFCC_COEFFS_LEN, MFCC_COEFFS_FIRST, 26, AUDIO_FRAME_LEN, 0.97, 1)
    for i in range(31):
        for j in range(512):
            dma_audio_buffer[j] = data[i*512+j]
        audio_sample_i += 512
        thread_kws_serv()
        
    for j in range(128):
        dma_audio_buffer[i] = data[i]
    thread_kws_serv()
    audio_sample_i = 0
    

import random

def main():
    # Generate array of 16000 random integers
    random_ints = [random.randint(-32768, 32767) for _ in range(16000)]

    # Call process_audio_data function with the generated random integers
    process_audio_data(random_ints)

if __name__ == "__main__":
    main()