#version 430 core

layout(local_size_x = 256) in;  // Workgroup size: 256 threads

struct Particle {
    vec4 position; // x, y, z, w
    vec4 velocity; // vx, vy, vz, padding
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};


uniform float deltaTime;
uniform vec3 boundingBox;

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= particles.length()) return;

    Particle p = particles[index];


    p.position.xyz += p.velocity.xyz * deltaTime;

    for (int i = 0; i < 3; i++) {
        if (p.position[i] < -boundingBox[i]) {
            p.position[i] = -boundingBox[i];
            p.velocity[i] *= -0.9;
        }
        if (p.position[i] > boundingBox[i]) {
            p.position[i] = boundingBox[i];
            p.velocity[i] *= -0.9;
        }
    }

    particles[index] = p;
}
