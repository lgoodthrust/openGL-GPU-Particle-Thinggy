#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 1) in float velocityMag;
out float vVelocityMag;
void main() {
    gl_Position = vec4(pos, 1.0);
    gl_PointSize = 2.0;
    vVelocityMag = velocityMag;
}