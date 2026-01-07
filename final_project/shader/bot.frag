#version 330 core

in vec3 worldPosition;
in vec3 worldNormal;
in vec4 vLightSpacePos;

out vec3 finalColor;

uniform vec3 lightPosition;
uniform vec3 lightIntensity;

uniform sampler2D uShadowMap;

uniform vec3 cameraPosition;
uniform vec3 fogColor;
uniform float fogStart;
uniform float fogEnd;

float shadowFactor(vec4 lightSpacePos, vec3 N, vec3 L) {
    vec3 ndc = lightSpacePos.xyz / lightSpacePos.w;
    vec3 sc = ndc * 0.5 + 0.5;

    if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0 || sc.z < 0.0 || sc.z > 1.0)
        return 1.0;

    float closest = texture(uShadowMap, sc.xy).r;
    float current = sc.z;

    float bias = max(0.0018 * (1.0 - dot(N, L)), 0.0006);
    return (current - bias > closest) ? 0.35 : 1.0;
}

void main() {
    vec3 N = normalize(worldNormal);
    vec3 Lvec = lightPosition - worldPosition;
    float dist2 = max(dot(Lvec, Lvec), 1e-4);
    vec3 L = normalize(Lvec);

    float NdotL = max(dot(N, L), 0.0);

    vec3 V = normalize(cameraPosition - worldPosition);
    vec3 H = normalize(L + V);

    float specPow = 64.0;
    float spec = pow(max(dot(N, H), 0.0), specPow);

    vec3 diffuse  = NdotL * (lightIntensity / dist2);
    vec3 specular = spec * (lightIntensity / dist2) * 0.35;

    float sh = shadowFactor(vLightSpacePos, N, L);

    vec3 v = (diffuse + specular) * sh;

    float dist = distance(cameraPosition, worldPosition);
    float fogFactor = clamp((dist - fogStart) / (fogEnd - fogStart), 0.0, 1.0);
    v = mix(v, fogColor, fogFactor);

    finalColor = pow(v, vec3(1.0 / 2.2));
}
