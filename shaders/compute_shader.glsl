#version 430 core

#define PARTICLE_COUNT 10000
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
uniform float repulsionForce = 0.01;

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
    particles[i].velocity.xyz = (particles[i].velocity.xyz + gravity * dt) * 0.999;

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