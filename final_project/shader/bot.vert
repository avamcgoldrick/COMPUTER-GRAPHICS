#version 330 core

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexNormal;
layout(location = 2) in vec2 vertexUV;
layout(location = 3) in vec4 vertexJointsFloat;
layout(location = 4) in vec4 vertexWeights;

out vec3 worldPosition;
out vec3 worldNormal;
out vec4 vLightSpacePos;

uniform mat4 MVP;       
uniform mat4 uModel;    
uniform mat4 uLightVP;   
uniform mat4 jointMatrices[100];

void main() {
    uvec4 j = uvec4(vertexJointsFloat);

    mat4 skinMat =
        vertexWeights.x * jointMatrices[j.x] +
        vertexWeights.y * jointMatrices[j.y] +
        vertexWeights.z * jointMatrices[j.z] +
        vertexWeights.w * jointMatrices[j.w];

    vec4 skinnedLocal = skinMat * vec4(vertexPosition, 1.0);
    vec4 wp = uModel * skinnedLocal;

    gl_Position = MVP * skinnedLocal;

    worldPosition = wp.xyz;

    mat3 M = mat3(uModel) * mat3(skinMat);
    worldNormal = normalize(M * vertexNormal);

    vLightSpacePos = uLightVP * wp;
}
