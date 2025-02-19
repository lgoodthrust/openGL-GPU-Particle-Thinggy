#version 330 core
in float vVelocityMag;
out vec4 color;
void main() {
    float speed = clamp(vVelocityMag / 10.0, 0.0, 1.0);
    color = mix(vec4(1.0, 1.0, 1.0, 0.25), vec4(1.0, 0.0, 0.0, 1.0), speed);
}