// Uses Declarative syntax to run commands inside a container.
pipeline {
    agent {
        kubernetes {
            inheritFrom 'default'
            yamlFile 'ci/pod/builder-build.yaml'
            defaultContainer 'kaniko'
        }
    }

    parameters{
        string(
            description: 'os(ubuntu20.04,centos7,ubuntu18.04)',
            name: 'os',
            defaultValue: 'ubuntu20.04'
        )
    }
     stages {
        stage ('Build'){
            steps {
                container('kaniko') {
                    dir ('knowhere') {
                        script {
                            sh 'ls -lah'
                            def date = sh(returnStdout: true, script: 'date +%Y%m%d').trim()
                            def gitShortCommit = sh(returnStdout: true, script: 'git rev-parse --short HEAD').trim()
                            def new_image="milvusdb/knowhere-gpu-build:amd64-${os}-${date}-${gitShortCommit}"
                            sh """
                            executor \
                            --registry-mirror="docker-nexus-ci.zilliz.cc"\
                            --insecure-registry="docker-nexus-ci.zilliz.cc" \
                            --dockerfile "ci/docker/builder/gpu/${os}/Dockerfile" \
                            --destination=${new_image} \
                            """
                        }
                    }
                }

            }
        }
    }

}
