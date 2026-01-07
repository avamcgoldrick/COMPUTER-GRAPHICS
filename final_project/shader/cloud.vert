#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;

uniform mat4 uMVP;
uniform mat4 uModel;

out vec2 vUV;
out vec3 worldPosition;

void main() {
    vUV = aUV;
    vec4 wp = uModel * vec4(aPos, 1.0);
    worldPosition = wp.xyz;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
