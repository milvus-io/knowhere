#pragma once

#include <optional>

namespace knowhere {

class FileOperator {
    /**
     * @brief Load a file to the local disk, so we can use stl lib to operate it.
     *
     * @param filename
     * @return false if any error, or return true.
     */
    virtual bool
    load_file(const std::string& filename) noexcept;

    /**
     * @brief Check if a file exists.
     *
     * @param filename
     * @return std::nullopt if any error, or return if the file exists.
     */
    virtual std::optional<bool>
    is_existed(const std::string& filename) noexcept;

    /**
     * @brief Create a file.
     *
     * @param filename
     * @return false if any error, or return true.
     */
    virtual bool
    create_file(const std::string& filename) noexcept;

    /**
     * @brief Delete a file.
     *
     * @param filename
     * @return false if any error, or return true.
     */
    virtual bool
    delete_file(const std::string& filename) noexcept;

    /**
     * @brief Rename a file.
     *
     * @param filename
     * @return false if any error, or return true.
     */
    virtual bool
    rename_file(const std::string& from_name, const std::string& to_name) noexcept;
};

}  // namespace knowhere
