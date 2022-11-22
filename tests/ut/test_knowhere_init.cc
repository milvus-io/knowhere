#include "catch2/catch_test_macros.hpp"
#include "knowhere/comp/knowhere_config.h"

TEST_CASE("Knowhere global config", "[init]") {
    knowhere::KnowhereConfig::SetBlasThreshold(16384);
    knowhere::KnowhereConfig::SetEarlyStopThreshold(0);
    knowhere::KnowhereConfig::SetLogHandler();
    knowhere::KnowhereConfig::SetStatisticsLevel(0);
}
