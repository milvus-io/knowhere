#pragma once

#include <unordered_set>

#include "knowhere/file_manager.h"
namespace knowhere {
/**
 * @brief LocalFileManager is used for test and placeholder purpose. It will not do anything to the file on disk.
 *
 * This class is not thread-safe.
 */
class LocalFileManager : public FileManager {
 public:
    auto
    LoadFile(const std::string& filename) noexcept -> bool override {
        return true;
    }

    auto
    AddFile(const std::string& filename) noexcept -> bool override {
        files.insert(filename);
        return true;
    }

    auto
    IsExisted(const std::string& filename) noexcept -> std::optional<bool> override {
        return std::make_optional<bool>(files.find(filename) != files.end());
    }

    auto
    RemoveFile(const std::string& filename) noexcept -> bool override {
        files.erase(filename);
        return true;
    }

 private:
    std::unordered_set<std::string> files;
};

}  // namespace knowhere
