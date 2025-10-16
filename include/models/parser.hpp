#pragma once

#include "models/base.hpp"
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

namespace neollm::model {

using json = nlohmann::json;

// Factory class to parse and load a model from disk.
class ModelParser {
public:
    // Parses model directory and returns a concrete Model instance.
    static std::unique_ptr<Model> parse(const std::string &model_path);

private:
    // Populates ModelConfig from parsed JSON.
    static void load_base_config(ModelConfig &config, const json &j);
    // Loads and parses config.json into a ModelConfig.
    static std::unique_ptr<ModelConfig> load_config(const std::string &config_path);

    // Loads model weights from files in the given path.
    static std::unique_ptr<ModelWeights> load_weights(const std::string &model_path);

    // Normalizes raw weight names (e.g., strips "model." prefix).
    static std::string map_weight_name(const std::string &raw_name);
};

} // namespace neollm::model