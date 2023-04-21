from conan.tools.microsoft import is_msvc, msvc_runtime_flag
from conan.tools.build import check_min_cppstd
from conan.tools.scm import Version
from conan.tools import files
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.errors import ConanInvalidConfiguration
from conans import tools
import os

required_conan_version = ">=1.55.0"


class KnowhereConan(ConanFile):
    name = "knowhere"
    description = "Knowhere is written in C++. It is an independent project that act as Milvus's internal core"
    topics = ("vector", "simd", "ann")
    url = "https://github.com/milvus-io/knowhere"
    homepage = "https://github.com/milvus-io/knowhere"
    license = "Apache-2.0"

    generators = "pkg_config"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_raft": [True, False],
        "with_asan": [True, False],
        "with_diskann": [True, False],
        "with_profiler": [True, False],
        "with_ut": [True, False],
        "with_benchmark": [True, False],
    }
    default_options = {
        "shared": True,
        "fPIC": False,
        "with_raft": False,
        "with_asan": False,
        "with_diskann": False,
        "with_profiler": False,
        "with_ut": False,
        "glog:with_gflags": False,
        "prometheus-cpp:with_pull": False,
        "with_benchmark": False,
    }

    exports_sources = (
        "src/*",
        "thirdparty/*",
        "tests/ut/*",
        "include/*",
        "CMakeLists.txt",
        "*.cmake",
        "conanfile.py",
    )

    @property
    def _minimum_cpp_standard(self):
        return 17

    @property
    def _minimum_compilers_version(self):
        return {
            "gcc": "8",
            "Visual Studio": "16",
            "clang": "6",
            "apple-clang": "10",
        }

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def requirements(self):
        self.requires("boost/1.81.0")
        self.requires("glog/0.6.0")
        self.requires("nlohmann_json/3.11.2")
        self.requires("openssl/1.1.1t")
        self.requires("prometheus-cpp/1.1.0")
        if self.options.with_ut:
            self.requires("catch2/3.3.1")
        if self.options.with_benchmark:
            self.requires("gtest/1.13.0")
            self.requires("hdf5/1.14.0")

    @property
    def _required_boost_components(self):
        return ["program_options"]

    def validate(self):
        if self.settings.compiler.get_safe("cppstd"):
            check_min_cppstd(self, self._minimum_cpp_standard)
        min_version = self._minimum_compilers_version.get(str(self.settings.compiler))
        if not min_version:
            self.output.warn(
                "{} recipe lacks information about the {} compiler support.".format(
                    self.name, self.settings.compiler
                )
            )
        else:
            if Version(self.settings.compiler.version) < min_version:
                raise ConanInvalidConfiguration(
                    "{} requires C++{} support. The current compiler {} {} does not support it.".format(
                        self.name,
                        self._minimum_cpp_standard,
                        self.settings.compiler,
                        self.settings.compiler.version,
                    )
                )

    def layout(self):
        cmake_layout(self)

    def generate(self):

        tc = CMakeToolchain(self)
        tc.variables["CMAKE_POSITION_INDEPENDENT_CODE"] = self.options.get_safe(
            "fPIC", True
        )
        # Relocatable shared lib on Macos
        tc.cache_variables["CMAKE_POLICY_DEFAULT_CMP0042"] = "NEW"
        # Honor BUILD_SHARED_LIBS from conan_toolchain (see https://github.com/conan-io/conan/issues/11840)
        tc.cache_variables["CMAKE_POLICY_DEFAULT_CMP0077"] = "NEW"

        cxx_std_flag = tools.cppstd_flag(self.settings)
        cxx_std_value = (
            cxx_std_flag.split("=")[1]
            if cxx_std_flag
            else "c++{}".format(self._minimum_cpp_standard)
        )
        tc.variables["CXX_STD"] = cxx_std_value
        if is_msvc(self):
            tc.variables["MSVC_LANGUAGE_VERSION"] = cxx_std_value
            tc.variables["MSVC_ENABLE_ALL_WARNINGS"] = False
            tc.variables["MSVC_USE_STATIC_RUNTIME"] = "MT" in msvc_runtime_flag(self)
        tc.variables["WITH_ASAN"] = self.options.with_asan
        tc.variables["WITH_DISKANN"] = self.options.with_diskann
        tc.variables["WITH_RAFT"] = self.options.with_raft
        tc.variables["WITH_PROFILER"] = self.options.with_profiler
        tc.variables["WITH_UT"] = self.options.with_ut
        tc.variables["WITH_BENCHMARK"] = self.options.with_benchmark
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        # files.apply_conandata_patches(self)
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        files.rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))
        files.rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "knowhere")
        self.cpp_info.set_property("cmake_target_name", "Knowhere::knowhere")
        self.cpp_info.set_property("pkg_config_name", "libknowhere")

        self.cpp_info.components["libknowhere"].libs = ["knowhere"]

        self.cpp_info.components["libknowhere"].requires = [
            "boost::program_options",
            "glog::glog",
            "prometheus-cpp::core",
            "prometheus-cpp::push",
        ]

        self.cpp_info.filenames["cmake_find_package"] = "knowhere"
        self.cpp_info.filenames["cmake_find_package_multi"] = "knowhere"
        self.cpp_info.names["cmake_find_package"] = "Knowhere"
        self.cpp_info.names["cmake_find_package_multi"] = "Knowhere"
        self.cpp_info.names["pkg_config"] = "libknowhere"
        self.cpp_info.components["libknowhere"].names["cmake_find_package"] = "knowhere"
        self.cpp_info.components["libknowhere"].names[
            "cmake_find_package_multi"
        ] = "knowhere"

        self.cpp_info.components["libknowhere"].set_property(
            "cmake_target_name", "Knowhere::knowhere"
        )
        self.cpp_info.components["libknowhere"].set_property(
            "pkg_config_name", "libknowhere"
        )
