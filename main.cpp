#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sndfile.h>
#include <armadillo>
#include "MFCC.cpp"

using namespace std;

class SpeakerIdentifier 
{
private:

    MFCCExtractor extractor;
    // Speaker feature database
    struct SpeakerModel {
        string name;
        vector<vector<double>> featureVectors;
    };

    vector<SpeakerModel> speakerDatabase;

    // Compute cosine similarity between feature vectors
    double computeSimilarity(const vector<double>& vec1, const vector<double>& vec2) 
    {
        double dotProduct = 0.0;
        double norm1 = 0.0, norm2 = 0.0;

        for (size_t i = 0; i < vec1.size(); ++i) {
            dotProduct += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        return dotProduct / (sqrt(norm1) * sqrt(norm2));
    }

public:
    // Load .wav file and extract features
    vector<double> loadWavFile(const string& filepath) 
    {
        SF_INFO sfInfo;
        SNDFILE* infile = sf_open(filepath.c_str(), SFM_READ, &sfInfo);
        
        if (!infile) {
            throw runtime_error("Cannot open audio file: " + filepath);
        }

        vector<double> audioData(sfInfo.frames);
        sf_count_t count = sf_read_double(infile, audioData.data(), sfInfo.frames);
        sf_close(infile);

        return audioData;
    }

    // Train the speaker identification model
    void trainSpeaker(const string& speakerName, const string& trainingFilePath) 
    {
        // Load audio file
        vector<double> audioData = loadWavFile(trainingFilePath);
        
        // Extract features
        vector<double> features = extractor.extractMFCCFeatures(audioData);

        
        // Add to speaker model
        SpeakerModel model;
        model.name = speakerName;
        model.featureVectors.push_back(features);
        
        speakerDatabase.push_back(model);
    }

    // Identify the speaker of an input audio file
    string identifySpeaker(const string& testFilePath, double threshold = 0.9) // change to 0.996 // 
    {
        // Load audio file
        vector<double> audioData = loadWavFile(testFilePath);
        
        // Extract features
        vector<double> features = extractor.extractMFCCFeatures(audioData);

        // Compare against each speaker model
        string bestMatch = "Unknown";
        double highestSimilarity = -1.0;

        for (const auto& speakerModel : speakerDatabase) 
        {
            //printf("speakerDatabase = %s \n",speakerModel.name);
            for (const auto& modelFeatures : speakerModel.featureVectors) 
            {
                cout << "modelFeatures contents: ";
                for (int value : modelFeatures) 
                {
                    cout << value << " ";
                }
                cout << endl;

                double similarity = computeSimilarity(features, modelFeatures);
                cout << "similarity: " << fixed << setprecision(7) << similarity << endl;

                if (similarity > highestSimilarity) 
                {
                    highestSimilarity = similarity;
                    bestMatch = speakerModel.name;
                }
            }
        }

        // Apply confidence threshold
        return (highestSimilarity >= threshold) ? bestMatch : "Unknown";
    }
};

int main() 
{
    
    SpeakerIdentifier speaker_id;

    // Training phase
    speaker_id.trainSpeaker("gary", "./database/gary1.wav");
    speaker_id.trainSpeaker("gary", "./database/gary2.wav");
    speaker_id.trainSpeaker("John", "./database/1.hello.wav");
    speaker_id.trainSpeaker("John", "./database/1.name.wav");
    speaker_id.trainSpeaker("jessy","./database/2.hello.wav");
    speaker_id.trainSpeaker("jessy","./database/2.omar.wav");

    //read test wav from a text file 

    string wavFileName;

    // Open the text file
    ifstream file("input_wav.txt");
    if (!file.is_open()) {
        cerr << "Error: Unable to open file_list.txt" << endl;
        return 1;
    }
    // Read the first line from the text file
    getline(file, wavFileName);
    file.close();

    // Identification phase
    string identifiedSpeaker = speaker_id.identifySpeaker(wavFileName);
    cout << "Identified Speaker: " << identifiedSpeaker << endl;

    return 0;
}