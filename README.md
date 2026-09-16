## Setup

On Windows:

    py install 3.12
    py -3.12 -m venv .venv
    .venv\Scripts\activate

    pip install -r requirements.txt


## Building

    ./build.bat

Output is in build folder

## Mac

On Mac, we should convert the tflite model to an ML package.

    brew install python@3.12
    python3.12 -m venv .venv
    source .venv/bin/activate

    pip install coremltools tensorflow
    

    
