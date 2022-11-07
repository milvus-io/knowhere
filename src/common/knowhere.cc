#include "knowhere/knowhere.h"
namespace knowhere {

static int
InitLog() {
#if defined(CONSOLE_LOGGING)
    static auto console = spdlog::stderr_logger_mt("console");
#else
    try {
        auto max_size = 1048576 * 5;
        auto max_files = 3;
        static auto logfile = spdlog::rotating_logger_mt("filelog", "/tmp/knowhere.log", max_size, max_files);
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "init log file error.\n";
        return 1;
    }
#endif
    spdlog::get(LOG_KEY_NAME)->set_level(spdlog::level::warn);
    spdlog::get(LOG_KEY_NAME)->enable_backtrace(BACKTRACE_SIZE);
    return 0;
}

int __init_log_status__ = InitLog();

Index<IndexNode>
IndexFactory::Create(const std::string& name) {
    auto& func_mapping_ = MapInstance();
    assert(func_mapping_.find(name) != func_mapping_.end());
    KNOWHERE_INFO("create knowhere index {}", name);
    return func_mapping_[name]();
}

const IndexFactory&
IndexFactory::Register(const std::string& name, std::function<Index<IndexNode>()> func) {
    auto& func_mapping_ = MapInstance();
    func_mapping_[name] = func;
    return *this;
}

IndexFactory&
IndexFactory::Instance() {
    static IndexFactory factory;
    return factory;
}

IndexFactory::IndexFactory() {
}
IndexFactory::FuncMap&
IndexFactory::MapInstance() {
    static FuncMap func_map;
    return func_map;
}

}  // namespace knowhere
