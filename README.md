# Interactive Convolution & Correlation GUI

> A desktop application built with Python to provide an intuitive, animated visualization of discrete and continuous time convolution and correlation.

This tool is designed for students, educators, and engineers who want to build a deeper, visual understanding of how these fundamental signal processing operations work.

<img width="1895" height="1025" alt="image" src="https://github.com/user-attachments/assets/19f006d5-5ee7-41f1-b703-e65c69b18cb8" />



## Table of Contents
- [Key Features](#key-features)
- [Screenshots](#screenshots)
- [Technology Stack](#technology-stack)
- [Requirements](#requirements)
- [Installation & Usage](#installation--usage)
- [Code Structure Overview](#code-structure-overview)
- [Future Work](#future-work)
- [License](#license)

## Key Features

- **Dual-Mode Operation**: Seamlessly switch between **Discrete** and **Continuous** time signals. The UI dynamically adapts to show relevant input parameters for each mode.
- **Dual-Function Analysis**: Visualize both **Convolution** and **Correlation** to easily compare and contrast the two operations.
- **Three-Plot Visualization**: A clear, vertically-structured layout shows:
    1.  The two original input signals.
    2.  The dynamic sliding window, showing the interaction between the signals frame by frame.
    3.  The real-time construction of the output signal, synchronized with the animation.
- **Real-time Overlap Highlighting**: The overlapping area of the two signals in the sliding window is shaded green, visually representing the point-wise multiplication at the core of the operation.
- **Interactive Speed Controls**: Fine-tune the animation speed with an **FPS slider** for smooth playback and a **Frame Skip** input for rapidly moving through the timeline.
- **Robust Error Handling**: The application provides user-friendly warning dialogs for invalid inputs instead of crashing.

## Screenshots



**Main Interface (Discrete Convolution)**

<img width="1904" height="1025" alt="image" src="https://github.com/user-attachments/assets/1eb514c7-bed4-4074-97b7-1c06bec4d3ce" />
<img width="1909" height="1026" alt="image" src="https://github.com/user-attachments/assets/bb8431c5-d541-49b9-b966-2ba24e483cf5" />
<img width="1901" height="1022" alt="image" src="https://github.com/user-attachments/assets/a73e3166-038a-4905-a689-72331dc33291" />



**Continuous Correlation Animation**
<img width="1914" height="1013" alt="image" src="https://github.com/user-attachments/assets/d0672951-b66e-4b9b-a8d1-84282a302e1e" />
<img width="1903" height="1016" alt="image" src="https://github.com/user-attachments/assets/e0396193-9e32-4c50-9cf5-337ba87bb206" />
<img width="1904" height="1025" alt="image" src="https://github.com/user-attachments/assets/fa6ccfeb-b60b-4e5a-b3f8-8762715c2dca" />



**Continuous Correlation Animation**

<img width="1914" height="1035" alt="image" src="https://github.com/user-attachments/assets/bd33c885-5f92-4837-a2db-d22b55e8709a" />

**Discrete Correlation Animation**

<img width="1901" height="1020" alt="image" src="https://github.com/user-attachments/assets/20f0b8f8-ab1c-468c-bf31-765e8e67f224" />
<img width="1908" height="1026" alt="image" src="https://github.com/user-attachments/assets/c2e507fa-e086-4d9d-82da-2ecf7e0a77e3" />



## Technology Stack

- **Language**: Python 3
- **GUI Framework**: PyQt5
- **Numerical Computation**: NumPy
- **Plotting**: Matplotlib

## Requirements

To run this application, you will need Python 3 and the following libraries:
- `PyQt5`
- `numpy`
- `matplotlib`

## Installation & Usage

Follow these steps to get the application running on your local machine.

**1. Clone the Repository**
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

**2. Create a Virtual Environment (Recommended)**
It is highly recommended to create a virtual environment to manage project dependencies.
- **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **On Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

**3. Install Dependencies**
First, create a `requirements.txt` file with the following content:
```
PyQt5
numpy
matplotlib
```
Then, install the packages from this file:
```bash
pip install -r requirements.txt
```

**4. Run the Application**
Execute the main Python script to launch the GUI:
```bash
python gui.py
```

## Code Structure Overview

The entire application is contained within `gui.py`. Its architecture can be broken down into five functional parts:

- **1. Signal Representation (`DiscreteSignal`, `ContinuousSignal` classes)**:
  - These classes act as data structures to neatly bundle the properties of a signal (values, start index, type, etc.). This enforces a structured way of handling signal data throughout the application.

- **2. GUI Construction (`_setup_ui`, `_make_*_panel` methods)**:
  - These methods are responsible for initializing and arranging all the PyQt5 widgets. The main window layout is built using a combination of `QVBoxLayout`, `QHBoxLayout`, and `QGridLayout`.

- **3. Computational Engine (NumPy Integration)**:
  - The core mathematical work is delegated to NumPy. `np.convolve` and `np.correlate` are used for high-speed, accurate calculations for both discrete and continuous (sampled) signals.

- **4. Animation and Visualization (`QTimer`, `_update_frame` method)**:
  - The animation is driven by a `QTimer` to ensure a non-blocking, responsive GUI. The `_update_frame` method is the core animator, responsible for clearing and redrawing the Matplotlib canvas for every frame.

- **5. Event Handling & Control (`_on_*` methods, `_run_process` method)**:
  - This is the "nervous system" of the app. Methods like `_on_compute` are connected to widget signals (e.g., `button.clicked`). These trigger the main `_run_process` controller, which orchestrates the workflow from parsing user input to starting the animation.

## Future Work

- **Expand Signal Library**: Add more built-in signal types, such as sine waves, cosine waves, and exponential decays.
- **Allow Data Import**: Implement a feature to allow users to import their own signal data from a file (e.g., `.csv` or `.txt`).
- **Frequency-Domain View**: Add a fourth plot to show the Fourier Transform of the signals, visually demonstrating the Convolution Theorem (`Y(f) = X(f)H(f)`).
- **Theme/Style Options**: Add options for users to switch between light and dark themes for the UI.

## License

This project is licensed under the MIT License 
