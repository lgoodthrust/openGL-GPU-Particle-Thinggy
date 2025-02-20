#include <cmath>
#include <cstdlib>
#include <fstream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <limits>
#include <string>

 // Include ImGui and its GLFW/OpenGL3 bindings
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>

// ---------------------------------------------------
// Configuration
// ---------------------------------------------------
constexpr int INITIAL_PARTICLE_COUNT = 10000;
constexpr int MAX_PARTICLE_COUNT = 20000;  // Maximum particles we can simulate
constexpr int WORK_GROUP_SIZE = 256;
float        TIME_SCALE = 1.0f;

constexpr float GRAVITY_X = 0.0f;
constexpr float GRAVITY_Y = 0.0f;

// Mouse interaction
double mouseX = 0.0;
double mouseY = 0.0;
bool   mousePressed_l = false;
constexpr float MOUSE_FORCE = 10.0f;
constexpr float MOUSE_RANGE = 1.0f;
constexpr float REMOVE_RADIUS = 0.05f;

// Collision Box (world coordinates)
constexpr float BOX_MIN_X = -0.95f;
constexpr float BOX_MIN_Y = -0.95f;
constexpr float BOX_MIN_Z = -0.95f;
constexpr float BOX_MAX_X = 0.95f;
constexpr float BOX_MAX_Y = 0.95f;
constexpr float BOX_MAX_Z = 0.95f;
constexpr float RESTITUTION = 1.0f;
constexpr float FRICTION = 0.0f;
constexpr float REPULSION_STRENGTH = 1.0f;

// ---------------------------------------------------
// Particle Structure
// ---------------------------------------------------
// Must match layout in compute_shader.glsl
struct Particle {
    float x, y, z, _pad1;           // position (vec4)
    float vx, vy, vz, velocityMag;  // velocity (vec4)
    int   flagActive;               // 1 = active, 0 = inactive
    int   pad0;
    int   pad1;
    int   pad2;
};

// ---------------------------------------------------
// Global Data
// ---------------------------------------------------

// We preallocate MAX_PARTICLE_COUNT elements.
//  - 'activeCount' tracks how many are currently in use.
std::vector<Particle> gParticles;
int activeCount = INITIAL_PARTICLE_COUNT; // how many are active

// GLSL Programs & Buffers
GLuint particleBuffer = 0;
GLuint computeShaderProgram = 0;
GLuint renderShaderProgram = 0;
GLuint vao = 0;
GLuint gridBuffer = 0;
GLuint gridCounterBuffer = 0;

// Spatial grid parameters (must match shader)
constexpr int GRID_SIZE = 3;      // Number of grid cells per axis
constexpr int MAX_NEIGHBORS = 28;     // Max particles per grid cell
constexpr float CELL_SIZE = 1.0f;   // Cell size (must match shader)

// Mode flags
bool createMode = false;  // If true, click spawns new particles
bool removeMode = false;  // If true, click removes nearest particle
bool forceMode = false;  // If true, compute shader applies mouse force

// ---------------------------------------------------
// Shader Reading & Compilation
// ---------------------------------------------------
std::string readFile(const char* filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Failed to open file: ") + filePath);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
        std::cerr << "Shader Compilation Failed:\n" << infoLog << std::endl;
        return 0;
    }
    return shader;
}

// ---------------------------------------------------
// initShaders - sets up both compute and render pipeline
// ---------------------------------------------------
void initShaders() {
    // Adjust file paths if needed
    std::string compSrc = readFile("shaders/compute_shader.glsl");
    std::string vertSrc = readFile("shaders/vertex_shader.glsl");
    std::string fragSrc = readFile("shaders/fragment_shader.glsl");

    const char* compSource = compSrc.c_str();
    const char* vertSource = vertSrc.c_str();
    const char* fragSource = fragSrc.c_str();

    // --- Compute Shader ---
    GLuint compShaderObj = compileShader(GL_COMPUTE_SHADER, compSource);
    if (compShaderObj == 0) {
        std::cerr << "Failed to compile Compute Shader!" << std::endl;
        return;
    }
    computeShaderProgram = glCreateProgram();
    glAttachShader(computeShaderProgram, compShaderObj);
    glLinkProgram(computeShaderProgram);
    {
        GLint linkSuccess;
        glGetProgramiv(computeShaderProgram, GL_LINK_STATUS, &linkSuccess);
        if (!linkSuccess) {
            char infoLog[1024];
            glGetProgramInfoLog(computeShaderProgram, 1024, nullptr, infoLog);
            std::cerr << "Compute Shader Program Linking Failed:\n" << infoLog << std::endl;
            return;
        }
    }
    glDeleteShader(compShaderObj);

    //Render Shaders (Vertex + Fragment)
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertSource);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragSource);
    if (vs == 0 || fs == 0) {
        std::cerr << "Failed to compile Render Shaders!" << std::endl;
        return;
    }
    renderShaderProgram = glCreateProgram();
    glAttachShader(renderShaderProgram, vs);
    glAttachShader(renderShaderProgram, fs);
    glLinkProgram(renderShaderProgram);
    {
        GLint linkSuccess;
        glGetProgramiv(renderShaderProgram, GL_LINK_STATUS, &linkSuccess);
        if (!linkSuccess) {
            char infoLog[1024];
            glGetProgramInfoLog(renderShaderProgram, 1024, nullptr, infoLog);
            std::cerr << "Render Shader Program Linking Failed:\n" << infoLog << std::endl;
            return;
        }
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
}

// ---------------------------------------------------
// updateParticleBuffer
//   - Allocates the full buffer (for MAX_PARTICLE_COUNT).
//   - Uploads the current gParticles data to the GPU.
// ---------------------------------------------------
void updateParticleBuffer() {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleBuffer);
    // We always allocate enough for MAX_PARTICLE_COUNT.
    glBufferData(GL_SHADER_STORAGE_BUFFER,
        MAX_PARTICLE_COUNT * sizeof(Particle),
        gParticles.data(),
        GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particleBuffer);

    // Set up VAO attribute pointers
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, particleBuffer);

    // Position attribute: first 3 floats of Particle
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)0);
    glEnableVertexAttribArray(0);

    // Velocity magnitude attribute: at offset of velocityMag
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(Particle),
        (void*)offsetof(Particle, velocityMag));
    glEnableVertexAttribArray(1);
}

// ---------------------------------------------------
// initParticles
//   - Preallocates gParticles with size = MAX_PARTICLE_COUNT
//   - Randomizes the first INITIAL_PARTICLE_COUNT
//   - Marks the rest as inactive
// ---------------------------------------------------
void initParticles() {
    gParticles.resize(MAX_PARTICLE_COUNT);

    // Fill the first 'activeCount' particles with random positions/velocities
    for (int i = 0; i < activeCount; ++i) {
        Particle& p = gParticles[i];
        p.x = BOX_MIN_X + static_cast<float>(rand()) / RAND_MAX * (BOX_MAX_X - BOX_MIN_X);
        p.y = BOX_MIN_Y + static_cast<float>(rand()) / RAND_MAX * (BOX_MAX_Y - BOX_MIN_Y);
        p.z = BOX_MIN_Z + static_cast<float>(rand()) / RAND_MAX * (BOX_MAX_Z - BOX_MIN_Z);

        p.vx = (rand() % 20 - 10) / 100.0f;
        p.vy = (rand() % 20 - 10) / 100.0f;
        p.vz = (rand() % 20 - 10) / 100.0f;

        p.velocityMag = std::sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
        p.flagActive = 1;
        p.pad0 = p.pad1 = p.pad2 = 0;
    }

    // Mark the remaining as inactive
    for (int i = activeCount; i < MAX_PARTICLE_COUNT; ++i) {
        gParticles[i].flagActive = 0;
    }

    glGenBuffers(1, &particleBuffer);
    glGenVertexArrays(1, &vao);

    updateParticleBuffer();

    // Initialize grid buffers for neighbor lookups
    std::vector<int> gridData(GRID_SIZE * GRID_SIZE * GRID_SIZE * MAX_NEIGHBORS, -1);
    glGenBuffers(1, &gridBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
        gridData.size() * sizeof(int),
        gridData.data(),
        GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gridBuffer);

    std::vector<int> gridCounterData(GRID_SIZE * GRID_SIZE * GRID_SIZE, 0);
    glGenBuffers(1, &gridCounterBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridCounterBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
        gridCounterData.size() * sizeof(int),
        gridCounterData.data(),
        GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gridCounterBuffer);
}

// ---------------------------------------------------
// computeParticles
//   - Dispatches the compute shader for the active particles
// ---------------------------------------------------
void computeParticles(float dt) {
    // Clear grid data
    int clearVal = -1;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridBuffer);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED_INTEGER, GL_INT, &clearVal);

    int zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridCounterBuffer);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED_INTEGER, GL_INT, &zero);

    float normalizedMouseX = (mouseX / 800.0f) * 2.0f - 1.0f;
    float normalizedMouseY = 1.0f - (mouseY / 800.0f) * 2.0f;

    glUseProgram(computeShaderProgram);

    glUniform1f(glGetUniformLocation(computeShaderProgram, "dt"), dt);
    glUniform3f(glGetUniformLocation(computeShaderProgram, "gravity"), GRAVITY_X, GRAVITY_Y, 0.0f);
    glUniform3f(glGetUniformLocation(computeShaderProgram, "mousePos"),
        normalizedMouseX, normalizedMouseY, 0.0f);
    glUniform1i(glGetUniformLocation(computeShaderProgram, "mousePressed_l"),
        (forceMode && mousePressed_l) ? 1 : 0);
    glUniform1f(glGetUniformLocation(computeShaderProgram, "mouseForce"), MOUSE_FORCE);
    glUniform1f(glGetUniformLocation(computeShaderProgram, "mouseRange"), MOUSE_RANGE);
    glUniform3f(glGetUniformLocation(computeShaderProgram, "boxMin"), BOX_MIN_X, BOX_MIN_Y, BOX_MIN_Z);
    glUniform3f(glGetUniformLocation(computeShaderProgram, "boxMax"), BOX_MAX_X, BOX_MAX_Y, BOX_MAX_Z);
    glUniform1f(glGetUniformLocation(computeShaderProgram, "restitution"), RESTITUTION);
    glUniform1f(glGetUniformLocation(computeShaderProgram, "friction"), FRICTION);
    glUniform1f(glGetUniformLocation(computeShaderProgram, "repulsionStrength"), REPULSION_STRENGTH);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particleBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gridBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gridCounterBuffer);

    // Use activeCount as the number of particles to simulate
    int particleCount = activeCount;
    glDispatchCompute((particleCount + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ---------------------------------------------------
// renderParticles
//   - Uses the render shader to draw the active particles
// ---------------------------------------------------
void renderParticles() {
    glUseProgram(renderShaderProgram);
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, activeCount);
}

// ---------------------------------------------------
// getMouseSimPosition
//   - Convert window coords to simulation coords
// ---------------------------------------------------
void getMouseSimPosition(float& simX, float& simY) {
    simX = (mouseX / 800.0) * 2.0f - 1.0f;
    simY = 1.0f - (mouseY / 800.0) * 2.0f;
}

// ---------------------------------------------------
// addParticleAtMouse
//   - Puts a new particle in the next available slot
//   - Minimizes data reuploads
// ---------------------------------------------------
void addParticleAtMouse() {
    if (activeCount >= MAX_PARTICLE_COUNT) return;

    float simX, simY;
    getMouseSimPosition(simX, simY);

    Particle newParticle;
    newParticle.x = simX;
    newParticle.y = simY;
    newParticle.z = 0.0f;

    newParticle.vx = (rand() % 20 - 10) / 100.0f;
    newParticle.vy = (rand() % 20 - 10) / 100.0f;
    newParticle.vz = (rand() % 20 - 10) / 100.0f;
    newParticle.velocityMag = std::sqrt(newParticle.vx * newParticle.vx +
        newParticle.vy * newParticle.vy +
        newParticle.vz * newParticle.vz);
    newParticle.flagActive = 1;
    newParticle.pad0 = newParticle.pad1 = newParticle.pad2 = 0;

    std::cout << "Adding particle at (" << simX << ", " << simY << ")\n";

    gParticles[activeCount] = newParticle;

    // Update only the new slot on the GPU
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleBuffer);
    size_t offset = activeCount * sizeof(Particle);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, sizeof(Particle),
        &gParticles[activeCount]);

    ++activeCount;
}

// ---------------------------------------------------
// removeParticleAtMouse
//   - Reads back the first 'activeCount' particles
//   - Finds the nearest to the mouse
//   - Shifts everything after it left by one
//   - Reuploads only the shifted region
// ---------------------------------------------------
void removeParticleAtMouse() {
    if (activeCount == 0) return;

    float simX, simY;
    getMouseSimPosition(simX, simY);

    // Read current GPU states into gParticles array
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
        activeCount * sizeof(Particle),
        gParticles.data());

    int   targetIndex = -1;
    float minDistSq = std::numeric_limits<float>::max();

    // Among the first activeCount, find the one nearest to the mouse
    for (int i = 0; i < activeCount; ++i) {
        float dx = gParticles[i].x - simX;
        float dy = gParticles[i].y - simY;
        float distSq = dx * dx + dy * dy;
        if (distSq < minDistSq) {
            minDistSq = distSq;
            targetIndex = i;
        }
    }

    // If within REMOVE_RADIUS, remove it by shifting
    if (targetIndex != -1 && minDistSq < (REMOVE_RADIUS * REMOVE_RADIUS)) {
        std::cout << "Removing particle at index " << targetIndex << "\n";

        for (int i = targetIndex; i < activeCount - 1; ++i) {
            gParticles[i] = gParticles[i + 1];
        }
        --activeCount;

        // Reupload only the shifted region
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleBuffer);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
            targetIndex * sizeof(Particle),
            (activeCount - targetIndex) * sizeof(Particle),
            &gParticles[targetIndex]);

        // Mark last slot as inactive
        gParticles[activeCount].flagActive = 0;
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
            activeCount * sizeof(Particle),
            sizeof(Particle),
            &gParticles[activeCount]);
    }
}

// ---------------------------------------------------
// GLFW Callbacks
// ---------------------------------------------------
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mousePressed_l = (action == GLFW_PRESS);
    }
}

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);
    if (ImGui::GetIO().WantCaptureMouse) return;

    mouseX = xpos;
    mouseY = ypos;
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    if (ImGui::GetIO().WantCaptureMouse) return;

    TIME_SCALE += static_cast<float>(yoffset) * 0.5f;
    if (TIME_SCALE < 0.5f) TIME_SCALE = 0.5f;
    if (TIME_SCALE > 3.0f) TIME_SCALE = 3.0f;
}

// ---------------------------------------------------
// Main Entry Point
// ---------------------------------------------------
int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!\n";
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(800, 800, "Optimized GPU Particles", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window!\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD!\n";
        return -1;
    }
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Initialize the simulation
    initShaders();
    initParticles();

    // Set up GLFW callbacks
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);

    float lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        float currentTime = glfwGetTime();
        float deltaTime = (currentTime - lastTime) * TIME_SCALE;
        lastTime = currentTime;

        glClear(GL_COLOR_BUFFER_BIT);

        // If the left mouse is pressed, spawn or remove particles.
        if (mousePressed_l) {
            if (createMode) {
                addParticleAtMouse();
            }
            else if (removeMode) {
                removeParticleAtMouse();
            }
        }

        // Update and render particles
        computeParticles(deltaTime);
        renderParticles();

        // Draw collision box
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glBegin(GL_QUADS);
        // Front face
        glVertex3f(-0.95f, -0.95f, -0.95f);
        glVertex3f(0.95f, -0.95f, -0.95f);
        glVertex3f(0.95f, 0.95f, -0.95f);
        glVertex3f(-0.95f, 0.95f, -0.95f);
        // Back face
        glVertex3f(-0.95f, -0.95f, 0.95f);
        glVertex3f(0.95f, -0.95f, 0.95f);
        glVertex3f(0.95f, 0.95f, 0.95f);
        glVertex3f(-0.95f, 0.95f, 0.95f);
        // Left face
        glVertex3f(-0.95f, -0.95f, -0.95f);
        glVertex3f(-0.95f, 0.95f, -0.95f);
        glVertex3f(-0.95f, 0.95f, 0.95f);
        glVertex3f(-0.95f, -0.95f, 0.95f);
        // Right face
        glVertex3f(0.95f, -0.95f, -0.95f);
        glVertex3f(0.95f, 0.95f, -0.95f);
        glVertex3f(0.95f, 0.95f, 0.95f);
        glVertex3f(0.95f, -0.95f, 0.95f);
        // Top face
        glVertex3f(-0.95f, 0.95f, -0.95f);
        glVertex3f(0.95f, 0.95f, -0.95f);
        glVertex3f(0.95f, 0.95f, 0.95f);
        glVertex3f(-0.95f, 0.95f, 0.95f);
        // Bottom face
        glVertex3f(-0.95f, -0.95f, -0.95f);
        glVertex3f(0.95f, -0.95f, -0.95f);
        glVertex3f(0.95f, -0.95f, 0.95f);
        glVertex3f(-0.95f, -0.95f, 0.95f);
        glEnd();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        // Begin a new ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Position & size for ImGui window
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::SetNextWindowSize(ImVec2(300, 150));
        ImGui::Begin("Menu");
        if (ImGui::Checkbox("Mouse Force Mode", &forceMode)) {}
        if (ImGui::Button("Reset Particles")) {
            activeCount = INITIAL_PARTICLE_COUNT;
            initParticles();
            std::cout << "Reset Particles button clicked!\n";
        }
        if (ImGui::Checkbox("Mouse Create Mode", &createMode)) {
            if (createMode) removeMode = false;
        }
        if (ImGui::Checkbox("Mouse Remove Mode", &removeMode)) {
            if (removeMode) createMode = false;
        }
        // Particle counter display
        ImGui::Text("Particles: %d / %d", activeCount, MAX_PARTICLE_COUNT);
        ImGui::End();

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers, poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Cleanup GL objects
    glDeleteProgram(computeShaderProgram);
    glDeleteProgram(renderShaderProgram);
    glDeleteBuffers(1, &particleBuffer);
    glDeleteBuffers(1, &gridBuffer);
    glDeleteBuffers(1, &gridCounterBuffer);
    glDeleteVertexArrays(1, &vao);

    // Cleanup GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
