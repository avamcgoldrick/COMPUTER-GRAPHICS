#version 330 core
layout(location=0) in vec3 aPos;

out vec3 vDir;

uniform mat4 uVP;

void main() {
    vDir = aPos; // direction for cubemap lookup
    vec4 p = uVP * vec4(aPos, 1.0);
    // keep at far depth
    gl_Position = vec4(p.xy, p.w, p.w);
}
