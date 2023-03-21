int total_timeout_minutes = 60*2
def knowhere_wheel=''
pipeline {
    agent {
        kubernetes {
            inheritFrom 'default'
            yamlFile 'ci/pod/ut-gpu.yaml'
            defaultContainer 'main'
        }
    }

    options {
        timeout(time: total_timeout_minutes, unit: 'MINUTES')
        buildDiscarder logRotator(artifactDaysToKeepStr: '30')
        parallelsAlwaysFailFast()
        disableConcurrentBuilds(abortPrevious: true)
        preserveStashes(buildCount: 10)
    }
    stages {
        stage("UT"){

            steps {
                container("build"){
                    script{
                        def date = sh(returnStdout: true, script: 'date +%Y%m%d').trim()
                        def gitShortCommit = sh(returnStdout: true, script: "echo ${env.GIT_COMMIT} | cut -b 1-7 ").trim()
                        version="${env.CHANGE_ID}.${date}.${gitShortCommit}"
                        sh "apt-get update || true"
                        sh "apt-get install dirmngr -y"
                        sh "apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 42D5A192B819C5DA"
                        sh "apt-get install build-essential libopenblas-dev ninja-build git -y"
                        sh "git config --global --add safe.directory '*'"
                        sh "git submodule update --recursive --init"
                        sh "mkdir build"
                        sh "cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_DISKANN=ON -DWITH_UT=ON -DWITH_ASAN=ON -DUSE_CUDA=ON \
                              && make -j8 \
                              && ./tests/ut/knowhere_tests"
                    }
                }
            }
        }
    }
}
