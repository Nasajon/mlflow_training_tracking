.DEFAULT_GOAL := build

build:
	docker build -t mlflow_training_tracking:0.2.0 -t mlflow_training_tracking:latest .