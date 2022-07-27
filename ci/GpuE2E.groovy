int total_timeout_minutes = 60
def knowhere_wheel=''
pipeline {
     agent {
        kubernetes {
            inheritFrom 'default'
            defaultContainer 'jnlp'
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
        stage("Build"){
                 agent {
                    kubernetes {
                        inheritFrom 'default'
                        yamlFile 'ci/pod/gpu-build.yaml'
                        defaultContainer 'main'
                    }
                }

            steps {
                script{
                    def date = sh(returnStdout: true, script: 'date +%Y%m%d').trim()
                    def gitShortCommit = sh(returnStdout: true, script: "echo ${env.GIT_COMMIT} | cut -b 1-7 ").trim()  
                    version="${env.CHANGE_ID}.${date}.${gitShortCommit}"
                    sh "./build.sh -g -u -t Release"
                    knowhere_wheel="knowhere-${version}-cp38-cp38-linux_x86_64.whl"
                    sh "cd python  && VERSION=${version} python3 setup.py bdist_wheel"
                    dir('python'){
                      archiveArtifacts artifacts: "dist/${knowhere_wheel}", followSymlinks: false
                    }
                    // stash knowhere info for rebuild E2E Test only
                    sh "echo ${knowhere_wheel} > knowhere.txt"
                    stash includes: 'knowhere.txt', name: 'knowhereWheel'
                }
            }    
        }
        stage("Test"){
            agent {
                kubernetes {
                    inheritFrom 'default'
                    yamlFile 'ci/pod/gpu-e2e.yaml'
                    defaultContainer 'main'
                }
            }
             environment {
                PIP_TRUSTED_HOST="nexus-nexus-repository-manager.nexus"
                PIP_INDEX_URL="http://nexus-nexus-repository-manager.nexus:8081/repository/pypi-all/simple"
                PIP_INDEX="http://nexus-nexus-repository-manager.nexus:8081/repository/pypi-all/pypi"
                PIP_FIND_LINKS="http://nexus-nexus-repository-manager.nexus:8081/repository/pypi-all/pypi"
            }

            steps {
                script{
                    if ("${knowhere_wheel}"==''){
                        dir ("knowhereWheel"){
                            try{
                                unstash 'knowhereWheel'
                                knowhere_wheel=sh(returnStdout: true, script: 'cat knowhere.txt | tr -d \'\n\r\'')
                            }catch(e){
                                error "No knowhereWheel info remained ,please rerun build to build new package."
                            }
                        }
                    }
                    checkout([$class: 'GitSCM', branches: [[name: '*/main']], extensions: [], 
                    userRemoteConfigs: [[credentialsId: 'milvus-ci', url: 'https://github.com/milvus-io/knowhere-test.git']]])   
                    dir('tests'){
                      unarchive mapping: ["dist/${knowhere_wheel}": "${knowhere_wheel}"]
                      sh "ls -lah"
                      sh "nvidia-smi"
                      sh "pip3 install ${knowhere_wheel} \
                          && pip3 install -r requirements.txt --timeout 30 --retries 6  &&  pytest -v -m gpu"
                    }
                }
            }
            post{
                always {
                    script{
                        sh 'cp /tmp/knowhere_ci.log knowhere_ci.log'
                        archiveArtifacts artifacts: 'knowhere_ci.log', followSymlinks: false
                    }
                }
            }    
        }
        
    }
}
