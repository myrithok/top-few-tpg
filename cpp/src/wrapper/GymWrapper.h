#ifndef GYM_WRAPPER_H
#define GYM_WRAPPER_H

#include <random>

#include <gegelati.h>

#include "include/gym/gym.h"
#include "GymWrapper.h"

/**
 * LearningEnvironment to play any simulation of gym environment.
 * This uses the gym_binding class, which encodes the client-side of a http
 * binding with a python gym server. GymWrapper contains variables to both
 * describe the state to gegelati and to communicate with the server.
 *
 */
class GymWrapper : public Learn::LearningEnvironment {
protected:
    /// turn currently played, used to end the simulation (e.g. we run 100 steps)
    int turn;

    /// reward of the simulation
    float reward;

    /// name of the environment of sim (e.g. "MountainCar-v0")
    std::string envName;

    /// Contains the environment of the simulation
    boost::shared_ptr<Gym::Environment> env;

    /// Contains the current state of the game
    Data::PrimitiveTypeArray<float> state = Data::PrimitiveTypeArray<float>((210 * 160 * 3));

    /// Handles connection with Gym server
    boost::shared_ptr<Gym::Client> client;

    /// Randomness control
    Mutator::RNG rng;

    /// Initializes the env of simulation and returns size of action space
    virtual void initialize(std::string chosenEnv);

    /// nb of parameters from the gym output vector that we want to look at
    int parametersToObserve;

    /// Used to cause reset on "game over" state
    bool done;

public:
    /**
     * Constructor.
     *
     * @param chosenEnv name of the environment of sim (e.g. "MountainCar-v0").
     * @param actionSpaceSize number of allowed actions in this environment.
     * @param nbInputParameters parameters to take into consideration. Indeed,
     * the gym binding provides an array as input that is bigger than the real
     * input. As a consequence, we have to know the number n of inputs a given
     * environment should have to only read the first n inputs received.
     */
    GymWrapper(std::string chosenEnv, int actionSpaceSize, int nbInputParameters): LearningEnvironment(actionSpaceSize),
        parametersToObserve(nbInputParameters) {
        initialize(chosenEnv);
        this->reset(0);
    };

    /**
     * \brief Copy constructor for the GymWrapper.
     *
     * Default copy constructor, unused because all attributes are not trivially
     * copyable.
     */
    GymWrapper(const GymWrapper& other) = default;

    /// Destructor
    ~GymWrapper() {
    };

    /// Inherited via LearningEnvironment
    virtual void doAction(std::vector<uint64_t> actionIDs) override;

    /// Inherited via LearningEnvironment
    virtual void
    reset(size_t seed = 0,
          Learn::LearningMode mode = Learn::LearningMode::TRAINING) override;

    /// Inherited via LearningEnvironment
    virtual std::vector<std::reference_wrapper<const Data::DataHandler>>
    getDataSources() override;

    /**
     * Inherited from LearningEnvironment.
     *
     * The score will be directly given by the environment
     */
    double getScore() const override;

    /// Inherited via LearningEnvironment
    virtual bool isTerminal() const override;

    /// Inherited via LearningEnvironment
    virtual bool isCopyable() const override;

    /// Inherited via LearningEnvironment
    virtual LearningEnvironment* clone() const {
        return new GymWrapper(*this);
    };
};

#endif
