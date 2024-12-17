#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <fftw3.h>

using namespace std;

class MFCCExtractor {
private:
    // Mel filterbank parameters
    const int NUM_MFCC_COEFFS = 20;
    const int SAMPLE_RATE = 16000;
    const int FRAME_LENGTH = 400;  // 25ms at 16kHz
    const int FRAME_SHIFT = 160;   // 10ms shift
    const double PRE_EMPHASIS_ALPHA = 0.97;

    // Mel filterbank
    //vector<vector<double>> melFilterbank;

    // Compute Mel frequency from linear frequency
    double linearToMel(double hz) 
    {
        return 2595.0 * log10(1.0 + hz / 700.0);
    }

    // Compute linear frequency from Mel frequency
    double melToLinear(double mel) 
    {
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
    }

    // // Create Mel filterbank
    // void createMelFilterbank() 
    // {
    //     const int FFT_SIZE = FRAME_LENGTH * 2;
    //     const int NUM_FILTERS = 26;
    //     const double LOW_FREQ = 0;
    //     const double HIGH_FREQ = SAMPLE_RATE / 2.0;

    //     // Convert frequency limits to Mel scale
    //     double lowMel = linearToMel(LOW_FREQ);
    //     double highMel = linearToMel(HIGH_FREQ);

    //     // Generate Mel points
    //     vector<double> melPoints(NUM_FILTERS + 2);
    //     for (int i = 0; i < NUM_FILTERS + 2; ++i) 
    //     {
    //         double mel = lowMel + (highMel - lowMel) * i / (NUM_FILTERS + 1);
    //         melPoints[i] = mel;
    //     }

    //     // Convert Mel points back to linear frequencies
    //     vector<int> linearPoints(NUM_FILTERS + 2);
    //     for (int i = 0; i < NUM_FILTERS + 2; ++i) 
    //     {
    //         linearPoints[i] = static_cast<int>(
    //             floor((FFT_SIZE + 1) * melToLinear(melPoints[i]) / SAMPLE_RATE)
    //         );
    //     }

    //     // Create filterbank
    //     melFilterbank.resize(NUM_FILTERS);
    //     for (int m = 1; m <= NUM_FILTERS; ++m) {
    //         vector<double> filter(FFT_SIZE / 2 + 1, 0.0);
            
    //         for (int k = 0; k <= FFT_SIZE / 2; ++k) {
    //             int f_m_minus = linearPoints[m-1];
    //             int f_m_center = linearPoints[m];
    //             int f_m_plus = linearPoints[m+1];

    //             // Triangular filter
    //             if (k >= f_m_minus && k <= f_m_center) {
    //                 filter[k] = (k - f_m_minus) / (f_m_center - f_m_minus);
    //             }
    //             else if (k > f_m_center && k <= f_m_plus) {
    //                 filter[k] = (f_m_plus - k) / (f_m_plus - f_m_center);
    //             }
    //         }

    //         melFilterbank[m-1] = filter;
    //     }
    // }

    // Pre-emphasis to amplify high frequencies
    vector<double> preEmphasis(const vector<double>& signal) 
    {
        vector<double> emphasized(signal.size());
        emphasized[0] = signal[0];
        
        for (size_t i = 1; i < signal.size(); ++i) {
            emphasized[i] = signal[i] - PRE_EMPHASIS_ALPHA * signal[i-1];
        }
        
        return emphasized;
    }

    // Apply Hamming window
    vector<double> applyHammingWindow(const vector<double>& frame) 
    {
        vector<double> windowed(frame.size());
        for (size_t i = 0; i < frame.size(); ++i) {
            double multiplier = 0.54 - 0.46 * cos(2 * M_PI * i / (frame.size() - 1));
            windowed[i] = frame[i] * multiplier;
        }
        return windowed;
    }

public:
    MFCCExtractor() 
    {
        //nothing;
    }

    vector<double> extractMFCCFeatures(const vector<double>& audioSignal) 
    {
        // Pre-emphasis
        vector<double> signal = preEmphasis(audioSignal);

        // Containers for MFCC features
        vector<double> mfccFeatures(NUM_MFCC_COEFFS, 0.0);

        // Prepare FFT
        fftw_complex *in, *out;
        fftw_plan p;
        const int FFT_SIZE = FRAME_LENGTH * 2;

        in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * FFT_SIZE);
        out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * FFT_SIZE);
        p = fftw_plan_dft_1d(FFT_SIZE, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

        // Process frames
        int numFrames = (signal.size() - FRAME_LENGTH) / FRAME_SHIFT + 1;
        
        for (int frameStart = 0; frameStart < numFrames; ++frameStart) 
        {
            // Extract frame
            vector<double> frame(FRAME_LENGTH);
            for (int i = 0; i < FRAME_LENGTH; ++i) {
                frame[i] = signal[frameStart * FRAME_SHIFT + i];
            }

            // Apply Hamming window
            frame = applyHammingWindow(frame);

            // Prepare for FFT (zero-pad)
            fill_n(&in[0][0], FFT_SIZE * 2, 0.0);
            for (int i = 0; i < FRAME_LENGTH; ++i) {
                in[i][0] = frame[i];
            }

            // Compute FFT
            fftw_execute(p);

            // Compute power spectrum
            vector<double> powerSpectrum(FFT_SIZE / 2 + 1);
            for (int k = 0; k <= FFT_SIZE / 2; ++k) 
            {
                powerSpectrum[k] = out[k][0] * out[k][0] + out[k][1] * out[k][1];
            }

            /* don't need for now ! -> good for sensitivity to noise */

            // // Apply Mel filterbank
            // vector<double> melEnergies(melFilterbank.size(), 0.0);
            // for (size_t m = 0; m < melFilterbank.size(); ++m) 
            // {
            //     for (size_t k = 0; k < powerSpectrum.size(); ++k) 
            //     {
            //         melEnergies[m] += powerSpectrum[k] * melFilterbank[m][k];
            //     }
            //     melEnergies[m] = log(melEnergies[m] + 1e-10);
            // }

            // DCT to get MFCCs
            for (int n = 0; n < NUM_MFCC_COEFFS; ++n) 
            {
                double sum = 0.0;
                for (size_t m = 0; m < powerSpectrum.size(); ++m) 
                {
                    sum += powerSpectrum[m] * cos(n * M_PI * (m + 0.5) / powerSpectrum.size());
                }
                mfccFeatures[n] += sum;
            }
        }

        // Normalize and average over frames
        for (int i = 0; i < NUM_MFCC_COEFFS; ++i) 
        {
            mfccFeatures[i] /= numFrames;
        }

        // Cleanup
        fftw_destroy_plan(p);
        fftw_free(in);
        fftw_free(out);

        return mfccFeatures;
    }
};