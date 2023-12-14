#include <unordered_set>
#include <string>
#include <cfloat>

#include <gegelati.h>

#include "wrapper/GymWrapper.h"

int main() {
    int actionSpaceSize = 0;
    if (TOP_FEW) {
        std::cout << "Top-Few TPG\n";
        actionSpaceSize = 6;
    } else {
        std::cout << "Single TPG\n";
        actionSpaceSize = 18;
    }
    // Create the instruction set for programs
    Instructions::Set set;
    auto minus = [](double a, double b) -> double { return (double) a - (double) b; };
    auto add = [](double a, double b) -> double { return a + b; };
    auto max = [](double a, double b) -> double { return std::max(a, b); };
    auto modulo = [](double a, double b) -> double { return b != 0.0 ? fmod(a, b) : DBL_MIN; };
    auto cond = [](double a, double b) -> double { return a < b ? -a : a; };


    set.add(*(new Instructions::LambdaInstruction<double, double>(minus)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(add)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(max)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(modulo)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(cond)));


    // Set the parameters for the learning process.
    // (Controls mutations probability, program lengths, and graph size
    // among other things)
    // Loads them from "params.json" file
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson(ROOT_DIR "/params.json", params);
#ifdef NB_GENERATIONS
	params.nbGenerations = NB_GENERATIONS;
#endif // !NB_GENERATIONS


    // Instantiate the LearningEnvironment
    // add a "true" in the constructor args to swap to non-adversarial
    GymWrapper le("ALE/Defender-v5", actionSpaceSize, 100800);

    // Instantiate and init the learning agent
    Learn::LearningAgent la(le, set, params);
    la.init();

    // Adds a logger to the LA (to get statistics on learning) on std::cout
    auto logCout = *new Log::LABasicLogger(la);

    // Adds another logger that will log in a file
    std::ofstream o("log");
    auto logFile = *new Log::LABasicLogger(la, o);

    // Create an exporter for all graphs
    File::TPGGraphDotExporter dotExporter("out_000.dot", *la.getTPGGraph());


    // Train for NB_GENERATIONS generations
    for (int i = 0; i < params.nbGenerations; i++) {
        char buff[12];
        sprintf(buff, "out_%03d.dot", i);
        dotExporter.setNewFilePath(buff);
        dotExporter.print();
        /*std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex*> result;
        result = la.evaluateAllRoots(i, Learn::LearningMode::VALIDATION);*/
        la.trainOneGeneration(i);
    }

    // Keep best policy
    la.keepBestPolicy();
    dotExporter.setNewFilePath("out_best.dot");
    dotExporter.print();


    // cleanup
    for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
        delete (&set.getInstruction(i));
    }

    // if we want to test the best agent

    return 0;
}
