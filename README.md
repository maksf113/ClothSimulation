# ClothSim

ClothSim is a real-time, high-performance cloth simulation project that leverages the power of **CUDA** for parallel physics calculations and **OpenGL** for modern, hardware-accelerated rendering. The simulation employs **Verlet integration** to realistically model the behavior of the cloth under various forces, such as gravity and wind. A user-friendly graphical user interface, built with **ImGui**, allows for dynamic interaction with the simulation, enabling real-time adjustments to parameters like wind speed.

This project serves as a practical demonstration of integrating advanced physics simulation techniques with modern graphics programming, showcasing the advantages of GPU acceleration in achieving interactive and visually compelling results.

## Key Features

* **CUDA-Accelerated Physics**: The core of the cloth simulation, including the Verlet integration and force calculations, is implemented in CUDA to run on the GPU, enabling high-performance simulation of a large number of particles.
* **OpenGL Rendering**: The cloth and its environment are rendered using modern OpenGL, with features like texturing, lighting, and shadows to create a visually realistic scene.
* **Verlet Integration**: A stable and efficient numerical integration method is used to simulate the motion of the cloth particles, ensuring realistic and predictable behavior.
* **Interactive GUI Controls**: An intuitive graphical user interface, created with ImGui, allows users to control simulation parameters in real-time. Currently, this includes adjusting the wind's strength.
* **Shadow Mapping**: The scene includes dynamic shadows, rendered using shadow mapping, which adds depth and realism to the visualization.
* **Object-Oriented Design**: The C++ code is organized in a clean, object-oriented manner, making it modular, extensible, and easy to understand.

---

## Showcase

Here are some examples of the cloth simulation in action:

![](screenshots/gif.GIF)
![](screenshots/1.PNG)
![](screenshots/2.PNG)

---

## Installation Instructions

### Dependencies

To build and run this project, you will need the following dependencies installed on your system:

* **CMake** (version 3.20 or higher)
* **A C++17 compliant compiler** (e.g., GCC, Clang, MSVC)
* **NVIDIA CUDA Toolkit** (the project is configured for compute capability 5.0, but this can be adjusted in `CMakeLists.txt`)
* **GLFW** (for windowing and input)
* **GLEW** (for OpenGL extension loading)
* **GLM** (for OpenGL mathematics)
* **ImGui** (for the graphical user interface)
* **stb_image** (for image loading)

### Building the Project

1.  **Clone the repository:**
    ```bash
    git clone https://your-repository-url/ClothSim.git
    cd ClothSim
    ```

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Run CMake to configure the project:**
    ```bash
    cmake ..
    ```

4.  **Build the project:**
    On Linux or macOS:
    ```bash
    make
    ```
    On Windows (with Visual Studio):
    ```
    cmake --build .
    ```

5.  **Run the application:**
    The executable will be located in the `build` directory.
    ```bash
    ./ClothSim
    ```

---

## Project Structure

The project is organized as follows:

* **`src/`**: Contains all the C++, CUDA, and shader source code.
    * **`cuda/`**: CUDA kernels and helper classes for GPU processing.
    * **`graphics/`**: OpenGL-related classes, including the renderer, camera, shaders, and abstractions for OpenGL objects (VBO, VAO, FBO, etc.).
    * **`models/`**: Classes for the 3D models in the scene, such as the `Cloth` and `Floor`.
    * **`tools/`**: Utility classes for input management, GUI, and timing.
    * **`window/`**: The `Window` class, which manages the application window.
* **`textures/`**: Contains the texture files used in the simulation.
* **`CMakeLists.txt`**: The CMake build script for the project.