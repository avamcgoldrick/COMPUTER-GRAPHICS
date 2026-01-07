#version 330 core
in vec3 vDir;

uniform samplerCube uCube;

out vec4 FragColor;

void main() {
    FragColor = texture(uCube, normalize(vDir));
}
