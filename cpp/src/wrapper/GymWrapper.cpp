#include "GymWrapper.h"

void GymWrapper::initialize(std::string chosenEnv) {
    this->envName = chosenEnv;
    this->client = Gym::client_create("127.0.0.1", 5000);
    this->env = client->make(this->envName);
}

void GymWrapper::doAction(std::vector<uint64_t> actionIDs) {
    Gym::State s;
    // Calculate the final action input by mixing the top 3 reported actions
    uint64_t actionID = 0;
    if (TOP_FEW) {
        bool fire = false;
        bool up = false;
        bool right = false;
        bool left = false;
        bool down = false;
        for (int i; i < 3; i++) {
            switch (actionIDs[i]) {
                case 1:
                    fire = true;
                break;
                case 2:
                    up = true;
                break;
                case 3:
                    right = true;
                break;
                case 4:
                    left = true;
                break;
                case 5:
                    down = true;
                break;
            }
        }
        if (right && left) {
            right = false;
            left = false;
        }
        if (up && down) {
            up = false;
            down = false;
        }
        if (!fire) {
            if (left && !right && !down && !up) {
                actionID = 4;
            } else if (!left && right && !down && !up) {
                actionID = 3;
            } else if (!left && !right && down && !up) {
                actionID = 5;
            } else if (!left && !right && !down && up) {
                actionID = 2;
            } else if (left && !right && down && !up) {
                actionID = 9;
            } else if (left && !right && !down && up) {
                actionID = 7;
            } else if (!left && right && down && !up) {
                actionID = 8;
            } else if (!left && right && !down && up) {
                actionID = 6;
            }
        } else {
            if (!left && !right && !down && !up) {
                actionID = 1;
            } else if (left && !right && !down && !up) {
                actionID = 12;
            } else if (!left && right && !down && !up) {
                actionID = 11;
            } else if (!left && !right && down && !up) {
                actionID = 13;
            } else if (!left && !right && !down && up) {
                actionID = 10;
            } else if (left && !right && down && !up) {
                actionID = 17;
            } else if (left && !right && !down && up) {
                actionID = 15;
            } else if (!left && right && down && !up) {
                actionID = 16;
            } else if (!left && right && !down && up) {
                actionID = 14;
            }
        }
    } else {
        actionID = actionIDs[0];
    }
    std::vector<float> action({(float) actionID});
    env->step(action, false, &s);

    for (int i = 0; i < this->parametersToObserve; i++) {
        this->state.setDataAt(typeid(float), i, s.observation[i]);
    }
    this->reward += s.observation[0]; // add current position
    if (s.observation[0] > maxDistance) {
        maxDistance = s.observation[0];
    }
    this->turn++;
}

void GymWrapper::reset(size_t seed, Learn::LearningMode mode) {
    // Create seed from seed and mode
    size_t hash_seed = Data::Hash<size_t>()(seed) ^ Data::Hash<Learn::LearningMode>()(mode);
    this->rng.setSeed(hash_seed);

    // sets 0 for each input data
    for (int i = 0; i < state.getLargestAddressSpace(); i++) {
        this->state.setDataAt(typeid(float), i, 0.0);
    }
    this->turn = 0;
    this->reward = 0;
    this->maxDistance = -std::numeric_limits<float>::max();
    this->client.reset();


    Gym::State s;
    this->env->reset(&s);
    for (int i = 0; i < this->parametersToObserve; i++) {
        this->state.setDataAt(typeid(float), i, s.observation[i]);
    }
}

std::vector<std::reference_wrapper<const Data::DataHandler>> GymWrapper::getDataSources() {
    auto result = std::vector<std::reference_wrapper<const Data::DataHandler>>({this->state});
    return result;
}

bool GymWrapper::isTerminal() const {
    return false;
}

bool GymWrapper::isCopyable() const {
    return false;
}

/*
Learn::LearningEnvironment *GymWrapper::clone() const {
    return new GymWrapper(*this);
}*/

double GymWrapper::getScore() const {
    return this->maxDistance;
}
