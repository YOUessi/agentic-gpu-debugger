#include "vector_api.h"
#include "json.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
constexpr std::size_t kMaxInputBytes = 4 * 1024 * 1024;
constexpr std::size_t kMaxElements = 65536;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "the vector protocol requires IEEE 754 float32");

std::string read_input() {
    std::string input;
    std::array<char, 8192> buffer{};
    while (std::cin.good()) {
        std::cin.read(buffer.data(), buffer.size());
        const auto count = static_cast<std::size_t>(std::cin.gcount());
        if (count > kMaxInputBytes - input.size()) {
            throw std::runtime_error("stdin exceeds the 4 MiB limit");
        }
        input.append(buffer.data(), count);
    }
    if (!std::cin.eof()) {
        throw std::runtime_error("failed to read stdin");
    }
    // The parser treats a NUL as end-of-input. It is not JSON whitespace and
    // must not allow a second payload to be hidden after the first object.
    if (input.find('\0') != std::string::npos) {
        throw std::runtime_error("NUL bytes are not allowed in JSON input");
    }
    return input;
}

Json parse_input(const std::string& input) {
    std::set<std::string> keys;
    const auto callback = [&keys](int depth, Json::parse_event_t event, Json& value) {
        if ((event == Json::parse_event_t::object_start && depth != 0) ||
            (event == Json::parse_event_t::array_start && depth != 1)) {
            throw std::runtime_error("expected one object with flat numeric arrays");
        }
        if (event == Json::parse_event_t::key &&
            !keys.insert(value.get<std::string>()).second) {
            throw std::runtime_error("duplicate JSON key");
        }
        return true;
    };
    // parse(), unlike stream extraction, requires the whole input to be JSON.
    Json value = Json::parse(input, callback);
    if (!value.is_object() || value.size() != 3 || !value.contains("n") ||
        !value.contains("a") || !value.contains("b")) {
        throw std::runtime_error("expected exactly the keys n, a, and b");
    }
    return value;
}

std::vector<float> read_vector(const Json& values, std::size_t n) {
    if (!values.is_array() || values.size() != n) {
        throw std::runtime_error("input arrays must have length n");
    }
    std::vector<float> result;
    result.reserve(n);
    for (const auto& value : values) {
        if (!value.is_number()) {
            throw std::runtime_error("array elements must be numbers");
        }
        const double number = value.get<double>();
        if (!std::isfinite(number) ||
            std::abs(number) > static_cast<double>(std::numeric_limits<float>::max())) {
            throw std::runtime_error("array elements must be finite float32 values");
        }
        result.push_back(static_cast<float>(number));
    }
    return result;
}
}  // namespace

int main() {
    try {
        const Json input = parse_input(read_input());
        if (!input["n"].is_number_integer() || input["n"] < 1 ||
            input["n"] > kMaxElements) {
            throw std::runtime_error("n must be an integer between 1 and 65536");
        }
        const auto n = input["n"].get<std::size_t>();
        const auto a = read_vector(input["a"], n);
        const auto b = read_vector(input["b"], n);
        // Unwritten values must not accidentally look like successful zeros.
        std::vector<float> output(n, std::numeric_limits<float>::quiet_NaN());
        const int status = run_vector_add(a.data(), b.data(), output.data(), n);
        if (status != 0) {
            throw std::runtime_error("run_vector_add failed with status " + std::to_string(status));
        }
        for (const float value : output) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("kernel output contains a nonfinite value");
            }
        }
        const Json result{{"dtype", "float32"}, {"shape", {n}}, {"values", output}};
        const std::string encoded = result.dump();
        std::cout << encoded << '\n';
        std::cout.flush();
        if (!std::cout.good()) {
            throw std::runtime_error("failed to write stdout");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vector harness: " << error.what() << '\n';
        return 1;
    }
}
