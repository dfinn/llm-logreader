## Overview
A utility which uses a large language model to answer questions about a specified log file, using langchain and ChromaDb. 
It uses Ollama to run the llama3 model locally, and uses an ephemeral ChromaDb instance to store and query embeddings.

## Pre-requisites
- Python 3.12
- Ollama installed and running (see https://ollama.com/download)

## Installation
Install the required packages using pip. It's recommended to use a virtual environment to manage dependencies.
```commandline
pip install -r requirements.txt
```

Pull the required Ollama models:
```commandline
ollama pull llama3
ollama pull nomic-embed-text
```

### Usage
```commandline
python llm-logreader.py <log_file_path>
```

The log file will be read and split into lines, which will then be used to create embeddings for the large language model. 
The script will prompt you to enter a question about the log file, and it will then use the embeddings to find the most relevant lines from the log and feed them into the llm prompt.
The script will display the lines from the log file that are most relevant to your question, along with an answer based on that context.


### Example
Here's an example of running the script with a PyCharm log file and asking a question about it:

```commandline
python llm-logreader.py ~/Library/Logs/JetBrains/PyCharmCE2023.1/idea.log
Loading text from log file: /Users/dain/Library/Logs/JetBrains/PyCharmCE2023.1/idea.log
Loaded 2257059 characters from log file
Split 13210 lines from log file
Creating embeddings with 3 batches of size 5461
Processing batch 0...
Processing batch 1...
Processing batch 2...
Query: Why am I getting numpy related errors?
Processing...
---- Context -----
 [3636] 2024-07-03 16:25:26,997 [10211088136]   INFO - #c.j.p.s.s.PySkeletonGenerator$Run - Cache entry for numpy.core._umath_tests at '/Users/dain/Library/Caches/JetBrains/PyCharmCE2023.1/python_stubs/cache/83768e8b31bbb9ec5e50015d9ff41c62bc089f0b70096cbcf535e03e2891814c' indicates failed generation
 [11413] 2024-07-31 13:38:37,198 [336776794]   INFO - #c.j.p.s.s.PySkeletonGenerator$Run - Cache entry for numpy.core._umath_tests at '/Users/dain/Library/Caches/JetBrains/PyCharmCE2023.1/python_stubs/cache/83768e8b31bbb9ec5e50015d9ff41c62bc089f0b70096cbcf535e03e2891814c' indicates failed generation
 [6744] 2024-07-21 17:06:01,348 [1556978954]   INFO - #c.j.p.s.s.PySkeletonGenerator$Run - Cache entry for numpy.core._umath_tests at '/Users/dain/Library/Caches/JetBrains/PyCharmCE2023.1/python_stubs/cache/83768e8b31bbb9ec5e50015d9ff41c62bc089f0b70096cbcf535e03e2891814c' indicates failed generation
 [3448] 2024-07-03 16:16:44,939 [10210566078]   INFO - #c.j.p.s.s.PySkeletonGenerator$Run - Cache entry for numpy.core._umath_tests at '/Users/dain/Library/Caches/JetBrains/PyCharmCE2023.1/python_stubs/cache/83768e8b31bbb9ec5e50015d9ff41c62bc089f0b70096cbcf535e03e2891814c' indicates failed generation
 [5856] 2024-07-17 11:38:08,551 [1191706157]   INFO - #c.j.p.s.s.PySkeletonGenerator$Run - Cache entry for numpy.core._umath_tests at '/Users/dain/Library/Caches/JetBrains/PyCharmCE2023.1/python_stubs/cache/83768e8b31bbb9ec5e50015d9ff41c62bc089f0b70096cbcf535e03e2891814c' indicates failed generation
 [11221] 2024-07-31 13:37:00,345 [336679941]   INFO - #c.j.p.s.s.PySkeletonGenerator$Run - Cache entry for numpy.core._umath_tests at '/Users/dain/Library/Caches/JetBrains/PyCharmCE2023.1/python_stubs/cache/83768e8b31bbb9ec5e50015d9ff41c62bc089f0b70096cbcf535e03e2891814c' indicates failed generation
 [4830] 2024-07-16 10:28:35,486 [1101133092]   INFO - #c.j.p.s.s.PySkeletonGenerator$Run - Cache entry for numpy.core._umath_tests at '/Users/dain/Library/Caches/JetBrains/PyCharmCE2023.1/python_stubs/cache/83768e8b31bbb9ec5e50015d9ff41c62bc089f0b70096cbcf535e03e2891814c' indicates failed generation
 [3812] 2024-07-03 16:29:46,825 [10211347964]   INFO - #c.j.p.s.s.PySkeletonGenerator$Run - Cache entry for numpy.core._umath_tests at '/Users/dain/Library/Caches/JetBrains/PyCharmCE2023.1/python_stubs/cache/83768e8b31bbb9ec5e50015d9ff41c62bc089f0b70096cbcf535e03e2891814c' indicates failed generation
 [6031] 2024-07-17 11:38:32,247 [1191729853]   INFO - #c.j.p.s.s.PySkeletonGenerator$Run - Cache entry for numpy.core._umath_tests at '/Users/dain/Library/Caches/JetBrains/PyCharmCE2023.1/python_stubs/cache/83768e8b31bbb9ec5e50015d9ff41c62bc089f0b70096cbcf535e03e2891814c' indicates failed generation
 [5471] 2024-07-17 11:36:57,557 [1191635163]   INFO - #c.j.p.s.s.PySkeletonGenerator$Run - Cache entry for numpy.core._umath_tests at '/Users/dain/Library/Caches/JetBrains/PyCharmCE2023.1/python_stubs/cache/83768e8b31bbb9ec5e50015d9ff41c62bc089f0b70096cbcf535e03e2891814c' indicates failed generation

----- Answer -----
The log file suggests that you are experiencing issues with generating Python stubs for NumPy using PyCharm's built-in tool, PySkeletonGenerator. Specifically, the cache entries for `numpy.core._umath_tests` are indicating failed generations.

This issue might be due to a few reasons:

1. **Corrupted cache**: The cache entry is failing to generate, which could indicate that the cache itself has become corrupted. Try deleting the cache directory and then restarting PyCharm.
2. **Python version mismatch**: Make sure you're using the same Python version in your project as the one used by PyCharm's stub generation tool. If there's a mismatch, it might cause issues with generating stubs for NumPy.
3. **NumPy installation issue**: Ensure that NumPy is properly installed and configured on your system. A faulty or incomplete installation can lead to issues when trying to generate stubs.
4. **PyCharm configuration**: Check your PyCharm settings to ensure that the Python interpreter and project SDK are correctly configured.

To troubleshoot further, you can try the following:

1. Enable verbose logging for the PySkeletonGenerator by setting the `PYCHART_SKELETON_GENERATOR_VERBOSE` environment variable to `true`. This might provide more detailed information about the generation process.
2. Try generating stubs for a different Python module or package to see if the issue is specific to NumPy.

If none of these suggestions help, feel free to provide more details about your project and setup, and I'll do my best to assist you in resolving the issue!
```