#include <cmath>
#include <cstdlib>
#include <fstream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

// Include ImGui and its GLFW/OpenGL3 bindings.
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>

// ---------------------------------------------------
// Configuration
// ---------------------------------------------------
constexpr int PARTICLE_COUNT = 10000;
constexpr int WORK_GROUP_SIZE = 256;
float TIME_SCALE = 1.0f;

constexpr float GRAVITY_X = 0.0f;
constexpr float GRAVITY_Y = 0.0f;

// Mouse interaction (only left mouse is used)
double mouseX = 0.0, mouseY = 0.0;
bool mousePressed_l = false;
constexpr float MOUSE_FORCE = 10.0f;
constexpr float MOUSE_RANGE = 1.0f;

// Collision Box Parameters (world coordinates)
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
struct Particle {
    float x, y, z, _pad1;           // Position
    float vx, vy, vz, velocityMag;   // Velocity and magnitude
};

// ---------------------------------------------------
// OpenGL objects
// ---------------------------------------------------
GLuint particleBuffer = 0;
GLuint computeShaderProgram = 0;
GLuint renderShaderProgram = 0;
GLuint vao = 0;
GLuint gridBuffer = 0;
GLuint gridCounterBuffer = 0;

// Define spatial grid parameters (must match shader)
constexpr int GRID_SIZE = 3;        // Number of grid cells per axis
constexpr int MAX_NEIGHBORS = 28;    // Max particles per grid cell
constexpr float CELL_SIZE = 1.0f;    // Cell size (must match shader)

// ---------------------------------------------------
// Shader Compilation Helper
// ---------------------------------------------------
std::string readFile(const char* filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file.");
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

// Initialize Shaders (Compute + Render)
void initShaders() {
    std::string fileContent = readFile("shaders/compute_shader.glsl");
    const char* finalShaderSrc = fileContent.c_str();
    std::string vertexShaderCode = readFile("shaders/vertex_shader.glsl");
    const char* vertexShaderSrc = vertexShaderCode.c_str();
    std::string fragmentShaderCode = readFile("shaders/fragment_shader.glsl");
    const char* fragmentShaderSrc = fragmentShaderCode.c_str();

    // Compute Shader Program
    GLuint compShaderObj = compileShader(GL_COMPUTE_SHADER, finalShaderSrc);
    if (compShaderObj == 0) {
        std::cerr << "Failed to compile Compute Shader!" << std::endl;
        return;
    }
    computeShaderProgram = glCreateProgram();
    glAttachShader(computeShaderProgram, compShaderObj);
    glLinkProgram(computeShaderProgram);
    GLint linkSuccess;
    glGetProgramiv(computeShaderProgram, GL_LINK_STATUS, &linkSuccess);
    if (!linkSuccess) {
        char infoLog[1024];
        glGetProgramInfoLog(computeShaderProgram, 1024, nullptr, infoLog);
        std::cerr << "Compute Shader Program Linking Failed:\n" << infoLog << std::endl;
        return;
    }
    glDeleteShader(compShaderObj);

    // Render Shader Program
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    if (vs == 0 || fs == 0) {
        std::cerr << "Failed to compile Render Shaders!" << std::endl;
        return;
    }
    renderShaderProgram = glCreateProgram();
    glAttachShader(renderShaderProgram, vs);
    glAttachShader(renderShaderProgram, fs);
    glLinkProgram(renderShaderProgram);
    glGetProgramiv(renderShaderProgram, GL_LINK_STATUS, &linkSuccess);
    if (!linkSuccess) {
        char infoLog[1024];
        glGetProgramInfoLog(renderShaderProgram, 1024, nullptr, infoLog);
        std::cerr << "Shader Program Linking Failed:\n" << infoLog << std::endl;
        return;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
}

// Initialize Particle Buffer, Grid, and VAO
void initParticles() {
    std::vector<Particle> particles(PARTICLE_COUNT);
    for (auto& p : particles) {
        p.x = BOX_MIN_X + static_cast<float>(rand()) / RAND_MAX * (BOX_MAX_X - BOX_MIN_X);
        p.y = BOX_MIN_Y + static_cast<float>(rand()) / RAND_MAX * (BOX_MAX_Y - BOX_MIN_Y);
        p.z = BOX_MIN_Z + static_cast<float>(rand()) / RAND_MAX * (BOX_MAX_Z - BOX_MIN_Z);
        p.vx = (rand() % 20 - 10) / 100.0f;
        p.vy = (rand() % 20 - 10) / 100.0f;
        p.vz = (rand() % 20 - 10) / 100.0f;
        p.velocityMag = std::sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
    }
    glGenBuffers(1, &particleBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, particles.size() * sizeof(Particle),
        particles.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particleBuffer);

    std::vector<int> gridData(GRID_SIZE * GRID_SIZE * GRID_SIZE * MAX_NEIGHBORS, -1);
    glGenBuffers(1, &gridBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gridData.size() * sizeof(int),
        gridData.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gridBuffer);

    std::vector<int> gridCounterData(GRID_SIZE * GRID_SIZE * GRID_SIZE, 0);
    glGenBuffers(1, &gridCounterBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridCounterBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gridCounterData.size() * sizeof(int),
        gridCounterData.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gridCounterBuffer);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, particleBuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(Particle),
        (void*)offsetof(Particle, velocityMag));
    glEnableVertexAttribArray(1);
}

// Update (Compute) Particle Positions
void computeParticles(float dt) {
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
    glUniform3f(glGetUniformLocation(computeShaderProgram, "mousePos"), normalizedMouseX, normalizedMouseY, 0.0f);
    glUniform1i(glGetUniformLocation(computeShaderProgram, "mousePressed_l"), mousePressed_l);
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

    glDispatchCompute((PARTICLE_COUNT + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// Optional: Reset Particles (right mouse reset disabled)
void resetParticles() {
    std::vector<Particle> particles(PARTICLE_COUNT);
    for (auto& p : particles) {
        p.x = (rand() % 200 - 100) / 100.0f;
        p.y = (rand() % 200 - 100) / 100.0f;
        p.z = (rand() % 200 - 100) / 100.0f;
        p.vx = (rand() % 20 - 10) / 100.0f;
        p.vy = (rand() % 20 - 10) / 100.0f;
        p.vz = (rand() % 20 - 10) / 100.0f;
        p.velocityMag = std::sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, particles.size() * sizeof(Particle),
        particles.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particleBuffer);
}

// Render Particles
void renderParticles() {
    glUseProgram(renderShaderProgram);
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, PARTICLE_COUNT);
}

// Draw Collision Box (Wireframe)
void drawCollisionBox() {
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
}

// GLFW Callbacks that forward events to ImGui and update simulation only if mouse isn't captured
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    // If ImGui wants to capture the mouse, do not update simulation state.
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mousePressed_l = (action == GLFW_PRESS);
    }
}

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    mouseX = xpos;
    mouseY = ypos;
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    // Only update simulation scroll if mouse is not captured
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    TIME_SCALE += static_cast<float>(yoffset) * 0.1f;
    if (TIME_SCALE < 0.1f)
        TIME_SCALE = 0.1f;
    if (TIME_SCALE > 3.0f)
        TIME_SCALE = 3.0f;
}

// Main
int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(800, 800, "Optimized GPU Particles", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD!" << std::endl;
        return -1;
    }
    glEnable(GL_PROGRAM_POINT_SIZE);

    // ----------------- Initialize ImGui -----------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    // ----------------------------------------------------

    initShaders();
    initParticles();
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);

    float lastTime = glfwGetTime();
    float elapsedTime = 0.0f;
    while (!glfwWindowShouldClose(window)) {
        float currentTime = glfwGetTime();
        float deltaTime = (currentTime - lastTime) * TIME_SCALE;
        lastTime = currentTime;
        elapsedTime += deltaTime;
        glClear(GL_COLOR_BUFFER_BIT);

        computeParticles(deltaTime);
        renderParticles();
        drawCollisionBox();
        // (Right mouse reset is disabled)

        // ---------------- Begin ImGui Frame ----------------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Create a simple GUI window with clickable buttons.
        ImGui::Begin("Mouse Functions");
        if (ImGui::Button("Apply Force")) {
            std::cout << "Apply Force button clicked!" << std::endl;
            // Optionally toggle a simulation mode here.
        }
        if (ImGui::Button("Reset Particles")) {
            resetParticles();
            std::cout << "Reset Particles button clicked!" << std::endl;
        }
        ImGui::End();
        // ---------------- End ImGui Frame ------------------

        // Render the GUI on top of the scene.
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // ---------------- Cleanup ImGui ----------------
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    // -------------------------------------------------

    glDeleteProgram(computeShaderProgram);
    glDeleteProgram(renderShaderProgram);
    glDeleteBuffers(1, &particleBuffer);
    glDeleteBuffers(1, &gridBuffer);
    glDeleteBuffers(1, &gridCounterBuffer);
    glDeleteVertexArrays(1, &vao);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
