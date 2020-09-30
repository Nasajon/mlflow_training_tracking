import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlflow_training_tracking",  # Replace with your own username
    version="0.0.1",
    author="Nilton Duarte",
    author_email=["niltonduarte at nasajon.com.br",
                  "nilton.gduarte at gmail.com"],
    description="Software para treinar modelos de machine learning e fazer o tracking na plataforma MLflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nasajon/mlflow_training_tracking",
    packages=setuptools.find_packages(),
    classifiers=[],
    python_requires='>=3.6',
)
