pipeline {
    agent any

    environment {
        AWS_REGION = "us-east-1"
        ECR_REPO = "379515516143.dkr.ecr.us-east-1.amazonaws.com/devops-ai"
        IMAGE_TAG = "latest"
    }

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/mantasha0cloud/AI-devops_Assistant.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    docker build -t allenaira/devops_assistant:new .
                '''
            }
        }

        stage('Login to ECR') {
            steps {
                sh '''
                    aws ecr get-login-password --region $AWS_REGION | \
                    docker login --username AWS --password-stdin $ECR_REPO
                '''
            }
        }

        stage('Tag Image') {
            steps {
                sh '''
                    docker tag allenaira/devops_assistant:new $ECR_REPO:$IMAGE_TAG
                '''
            }
        }

        stage('Push to ECR') {
            steps {
                sh '''
                    docker push $ECR_REPO:$IMAGE_TAG
                '''
            }
        }
    }

    post {
        success {
            echo "Image successfully pushed to ECR!"
        }
        failure {
            echo "Pipeline Failed!"
        }
    }
}
