#include <vector>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// ---------------------------------------------------
// Configuration
// ---------------------------------------------------
constexpr int PARTICLE_COUNT = 100000;
constexpr int WORK_GROUP_SIZE = 256;
float TIME_SCALE = 1.0f;

constexpr float GRAVITY_X = 0.0f;
constexpr float GRAVITY_Y = 0.0f;

// Mouse interaction
double mouseX = 0.0, mouseY = 0.0;
bool mousePressed_l = false;
bool mousePressed_r = false;
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
// Compute Shader Source (Optimized)
// ---------------------------------------------------
const char* computeShaderSrc = R"(
#version 430 core

#define PARTICLE_COUNT 100000
#define GRID_SIZE 3
#define MAX_NEIGHBORS 28
#define CELL_SIZE 1.0

layout (local_size_x = 256) in;

struct Particle {
    vec4 position;
    vec4 velocity;
};

layout (std430, binding = 0) buffer Particles {
    Particle particles[];
};

layout (std430, binding = 1) buffer Grid {
    int gridCells[];
};

layout (std430, binding = 2) buffer GridCounter {
    int gridCounters[];
};

uniform float dt;
uniform vec3 gravity;
uniform vec3 mousePos;
uniform bool mousePressed_l;
uniform float mouseForce;
uniform float mouseRange;

// Collision box
uniform vec3 boxMin;
uniform vec3 boxMax;
uniform float restitution;
uniform float friction;
uniform float repulsionStrength;

// Particle interactions
uniform float repulsionRadius = 0.01;
uniform float repulsionForce = 1.0;

ivec3 getCell(vec3 pos) {
    return clamp(ivec3((pos - boxMin) / CELL_SIZE), ivec3(0), ivec3(GRID_SIZE - 1));
}

int flattenIndex(ivec3 cell) {
    return (cell.x * GRID_SIZE * GRID_SIZE + cell.y * GRID_SIZE + cell.z) * MAX_NEIGHBORS;
}

int gridCounterIndex(ivec3 cell) {
    return cell.x * GRID_SIZE * GRID_SIZE + cell.y * GRID_SIZE + cell.z;
}

// Precomputed neighbor offsets (27 total)
const ivec3 neighborOffsets[27] = ivec3[](
    ivec3(-1,-1,-1), ivec3(-1,-1,0), ivec3(-1,-1,1),
    ivec3(-1,0,-1),  ivec3(-1,0,0),  ivec3(-1,0,1),
    ivec3(-1,1,-1),  ivec3(-1,1,0),  ivec3(-1,1,1),
    ivec3(0,-1,-1),  ivec3(0,-1,0),  ivec3(0,-1,1),
    ivec3(0,0,-1),   ivec3(0,0,0),   ivec3(0,0,1),
    ivec3(0,1,-1),   ivec3(0,1,0),   ivec3(0,1,1),
    ivec3(1,-1,-1),  ivec3(1,-1,0),  ivec3(1,-1,1),
    ivec3(1,0,-1),   ivec3(1,0,0),   ivec3(1,0,1),
    ivec3(1,1,-1),   ivec3(1,1,0),   ivec3(1,1,1)
);

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= PARTICLE_COUNT) return;

    // Update velocity with gravity and damping in one step.
    particles[i].velocity.xyz = (particles[i].velocity.xyz + gravity * dt) * 0.99;

    // Compute spatial grid cell for this particle.
    ivec3 cell = getCell(particles[i].position.xyz);
    int cellBase = flattenIndex(cell);

    // Use atomic operation to get a safe index for grid insertion.
    int counter = atomicAdd(gridCounters[gridCounterIndex(cell)], 1);
    if (counter < MAX_NEIGHBORS) {
        gridCells[cellBase + counter] = int(i);
    }

    // Particle interactions: soft collisions and repulsion.
    vec3 collisionResponse = vec3(0.0);
    vec3 repulsion = vec3(0.0);
    int neighborCount = 0;

    // Loop over precomputed neighbor offsets.
    for (int idx = 0; idx < 27; idx++) {
        ivec3 neighborCell = clamp(cell + neighborOffsets[idx], ivec3(0), ivec3(GRID_SIZE - 1));
        int neighborCellBase = flattenIndex(neighborCell);
        for (int j = 0; j < MAX_NEIGHBORS; j++) {
            int neighborIdx = gridCells[neighborCellBase + j];
            if (neighborIdx < 0 || neighborIdx == int(i)) continue;

            vec3 dir = particles[i].position.xyz - particles[neighborIdx].position.xyz;
            float dist = length(dir);
            if (dist > 0.0001 && dist < repulsionRadius) {
                dir = normalize(dir);
                float strength = (1.0 - dist / repulsionRadius) * repulsionForce;
                repulsion += dir * strength;

                vec3 relativeVelocity = particles[i].velocity.xyz - particles[neighborIdx].velocity.xyz;
                float velocityAlongNormal = dot(relativeVelocity, dir);
                if (velocityAlongNormal < 0) {
                    float elasticity = 0.5;
                    float impulseStrength = -(1.0 + elasticity) * velocityAlongNormal * 0.5;
                    collisionResponse += dir * impulseStrength;
                }
                neighborCount++;
            }
        }
    }

    if (neighborCount > 0) {
        particles[i].velocity.xyz += (repulsion + collisionResponse) * dt;
    }

    // Update particle position.
    particles[i].position.xyz += particles[i].velocity.xyz * dt;
    particles[i].velocity.w = length(particles[i].velocity.xyz);

    // Apply mouse attraction if left mouse button is pressed.
    if (mousePressed_l) {
        vec3 direction = mousePos - particles[i].position.xyz;
        float distance = length(direction);
        if (distance > 0.0001) {
            direction = normalize(direction);
            float forceStrength = (1.0 + distance / mouseRange) * mouseForce;
            particles[i].velocity.xyz += direction * forceStrength * dt;
        }
    }

    // Collision with box boundaries.
    for (int j = 0; j < 3; j++) {
        float nextPos = particles[i].position[j] + particles[i].velocity[j] * dt;
        if (nextPos < boxMin[j]) {
            particles[i].position[j] = boxMin[j] + 0.01;
            particles[i].velocity[j] = -particles[i].velocity[j] * restitution;
            particles[i].velocity[(j + 1) % 3] *= (1.0 - friction);
            particles[i].velocity[(j + 2) % 3] *= (1.0 - friction);
            if (abs(particles[i].velocity[j]) < 0.01)
                particles[i].velocity[j] += 0.02;
        }
        else if (nextPos > boxMax[j]) {
            particles[i].position[j] = boxMax[j] - 0.01;
            particles[i].velocity[j] = -particles[i].velocity[j] * restitution;
            particles[i].velocity[(j + 1) % 3] *= (1.0 - friction);
            particles[i].velocity[(j + 2) % 3] *= (1.0 - friction);
            if (abs(particles[i].velocity[j]) < 0.01)
                particles[i].velocity[j] -= 0.02;
        }
    }
}
)";

// ---------------------------------------------------
// Render Shaders (Unchanged)
// ---------------------------------------------------
const char* vertexShaderSrc = R"(
#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in float velocityMag;
out float vVelocityMag;
void main() {
    gl_Position = vec4(pos, 1.0);
    gl_PointSize = 2.0;
    vVelocityMag = velocityMag;
}
)";

const char* fragmentShaderSrc = R"(
#version 330 core
in float vVelocityMag;
out vec4 color;
void main() {
    float speed = clamp(vVelocityMag / 10.0, 0.0, 1.0);
    color = mix(vec4(0.0, 0.0, 1.0, 0.25), vec4(1.0, 0.0, 0.0, 1.0), speed);
}
)";

// ---------------------------------------------------
// Shader Compilation Helper
// ---------------------------------------------------
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
// Initialize Shaders (Compute + Render)
// ---------------------------------------------------
void initShaders() {
    // Compute Shader
    GLuint compShaderObj = compileShader(GL_COMPUTE_SHADER, computeShaderSrc);
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
    renderShaderProgram = glCreateProgram();
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    if (vs == 0 || fs == 0) {
        std::cerr << "Failed to compile Render Shaders!" << std::endl;
        return;
    }
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

// ---------------------------------------------------
// Initialize Particle Buffer, Grid, and VAO
// ---------------------------------------------------
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

    // Initialize spatial hash grid (gridCells), set all to -1.
    std::vector<int> gridData(GRID_SIZE * GRID_SIZE * GRID_SIZE * MAX_NEIGHBORS, -1);
    glGenBuffers(1, &gridBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gridData.size() * sizeof(int),
        gridData.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gridBuffer);

    // Initialize grid counter buffer (one counter per grid cell).
    std::vector<int> gridCounterData(GRID_SIZE * GRID_SIZE * GRID_SIZE, 0);
    glGenBuffers(1, &gridCounterBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridCounterBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gridCounterData.size() * sizeof(int),
        gridCounterData.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gridCounterBuffer);

    // Create VAO for rendering.
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, particleBuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(Particle),
        (void*)offsetof(Particle, velocityMag));
    glEnableVertexAttribArray(1);
}

// ---------------------------------------------------
// Update (Compute) Particle Positions
// ---------------------------------------------------
void computeParticles(float dt) {
    // Clear grid buffer to -1.
    int clearVal = -1;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridBuffer);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED_INTEGER, GL_INT, &clearVal);
    // Clear grid counter buffer to 0.
    int zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridCounterBuffer);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED_INTEGER, GL_INT, &zero);

    // Convert mouse position to normalized coordinates.
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
    // Removed glFinish() to avoid CPU-GPU synchronization stalls.
}

// ---------------------------------------------------
// Optional: Reset Particles (Right-click)
// ---------------------------------------------------
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

// ---------------------------------------------------
// Render Particles
// ---------------------------------------------------
void renderParticles() {
    glUseProgram(renderShaderProgram);
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, PARTICLE_COUNT);
}

// ---------------------------------------------------
// Draw Collision Box (Wireframe)
// ---------------------------------------------------
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

// ---------------------------------------------------
// GLFW Callbacks
// ---------------------------------------------------
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mousePressed_l = (action == GLFW_PRESS);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        mousePressed_r = (action == GLFW_PRESS);
    }
}

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    mouseX = xpos;
    mouseY = ypos;
}

// Add the scroll callback:
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    // Adjust the time scale factor; sensitivity factor is 0.1 (adjust as needed)
    TIME_SCALE += static_cast<float>(yoffset) * 0.1f;
    // Clamp between 0.01 and 10.0
    if (TIME_SCALE < 0.1f)
        TIME_SCALE = 0.1f;
    if (TIME_SCALE > 3.0f)
        TIME_SCALE = 3.0f;
}

// ---------------------------------------------------
// Main
// ---------------------------------------------------
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
    initShaders();
    initParticles();
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    // Register the new scroll callback
    glfwSetScrollCallback(window, scrollCallback);

    float lastTime = glfwGetTime();
    float elapsedTime = 0.0f;
    while (!glfwWindowShouldClose(window)) {
        float currentTime = glfwGetTime();
        // Use the mutable timeScale instead of the fixed TIME_SCALE
        float deltaTime = (currentTime - lastTime) * TIME_SCALE;
        lastTime = currentTime;
        elapsedTime += deltaTime;
        glClear(GL_COLOR_BUFFER_BIT);
        computeParticles(deltaTime);
        renderParticles();
        drawCollisionBox();
        if (mousePressed_r) {
            resetParticles();
            elapsedTime = 0.0f;
        }
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    // Cleanup remains unchanged
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