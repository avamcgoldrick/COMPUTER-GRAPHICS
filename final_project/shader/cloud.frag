#version 330 core
in vec2 vUV;
in vec3 worldPosition;

uniform sampler2D ucolor;

uniform vec3 cameraPosition;
uniform vec3 fogColor;
uniform float fogStart;
uniform float fogEnd;

out vec4 FragColor;

void main() {
    vec4 c = texture(ucolor, vUV);
    float a = c.a;
    if (a < 0.05) discard;

    float dist = distance(cameraPosition, worldPosition);
    float fogFactor = clamp((dist - fogStart) / (fogEnd - fogStart), 0.0, 1.0);

    vec3 rgb = mix(c.rgb, fogColor, fogFactor);
    FragColor = vec4(rgb, a);
}
