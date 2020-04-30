import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name="deepasr",
    version="0.1.1",
    author="Sai Kumar Yava",
    author_email="saikumar.geek@gmail.com",
    description="Keras(Tensorflow) implementations of Automatic Speech Recognition",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/scionoftech/DeepAsr",
    include_package_data=True,
    packages=['deepasr'],
    keywords=['deepspeech', 'asr', 'speech recognition', 'speech to text'],
    license='GNU',
    install_requires=['tensorflow>=2.0', 'pandas', 'tables', 'scipy', 'librosa'],
    python_requires='>=3.6',
)
