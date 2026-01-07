#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <render/shader.h>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cassert>
#include <cstring>

#define BUFFER_OFFSET(i) ((char*)NULL + (i))

static GLFWwindow* window;
static int windowWidth = 1024;
static int windowHeight = 768;

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);

static glm::vec3 eye_center(0.0f, 150.0f, 800.0f);
static glm::vec3 lookat(0.0f, 150.0f, 0.0f);
static glm::vec3 up(0.0f, 1.0f, 0.0f);
static float FoV = 25.0f;
static float zNear = 0.1f;
static float zFar = 10000.0f;

static float camSpeed = 600.0f; 
static float turnSpeed = 1.6f;    
static float yaw = -1.57f;        
static float pitch = 0.0f;

static GLuint gShadowFBO = 0;
static GLuint gShadowTex = 0;
static const int SHADOW_RES = 2048;
static glm::mat4 gLightVP(1.0f);

static GLuint gCloudDepthProg = 0;
static GLuint gBotDepthProg = 0;
static GLint gCloudDepth_uLightVP = -1, gCloudDepth_uModel = -1;
static GLint gBotDepth_uLightVP = -1, gBotDepth_uModel = -1, gBotDepth_uJoints = -1;

static GLuint CompileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len);
        glGetShaderInfoLog(s, len, nullptr, log.data());
        std::cerr << "Shader compile error:\n" << log.data() << "\n";
    }
    return s;
}

static GLuint LinkProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len);
        glGetProgramInfoLog(p, len, nullptr, log.data());
        std::cerr << "Program link error:\n" << log.data() << "\n";
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

static void initShadowMap() {
    glGenFramebuffers(1, &gShadowFBO);

    glGenTextures(1, &gShadowTex);
    glBindTexture(GL_TEXTURE_2D, gShadowTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24,
        SHADOW_RES, SHADOW_RES, 0,
        GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float border[4] = { 1,1,1,1 };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);

    glBindFramebuffer(GL_FRAMEBUFFER, gShadowFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, gShadowTex, 0);

    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "Shadow FBO not complete!\n";

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void initDepthPrograms() {
    const char* depthFS = R"GLSL(
        #version 330 core
        void main() { }
    )GLSL";

    const char* cloudVS = R"GLSL(
        #version 330 core
        layout(location=0) in vec3 aPos;
        uniform mat4 uLightVP;
        uniform mat4 uModel;
        void main() {
            gl_Position = uLightVP * uModel * vec4(aPos, 1.0);
        }
    )GLSL";

    const char* botVS = R"GLSL(
        #version 330 core
        layout(location=0) in vec3 vertexPosition;
        layout(location=3) in vec4 vertexJointsFloat;
        layout(location=4) in vec4 vertexWeights;

        uniform mat4 uLightVP;
        uniform mat4 uModel;
        uniform mat4 jointMatrices[100];

        void main() {
            uvec4 j = uvec4(vertexJointsFloat);
            mat4 skinMat =
                vertexWeights.x * jointMatrices[j.x] +
                vertexWeights.y * jointMatrices[j.y] +
                vertexWeights.z * jointMatrices[j.z] +
                vertexWeights.w * jointMatrices[j.w];

            vec4 skinnedLocal = skinMat * vec4(vertexPosition, 1.0);
            vec4 worldPos = uModel * skinnedLocal;
            gl_Position = uLightVP * worldPos;
        }
    )GLSL";

    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, depthFS);

    GLuint cvs = CompileShader(GL_VERTEX_SHADER, cloudVS);
    gCloudDepthProg = LinkProgram(cvs, fs);
    gCloudDepth_uLightVP = glGetUniformLocation(gCloudDepthProg, "uLightVP");
    gCloudDepth_uModel = glGetUniformLocation(gCloudDepthProg, "uModel");

    GLuint bfs = CompileShader(GL_FRAGMENT_SHADER, depthFS); 
    GLuint bvs = CompileShader(GL_VERTEX_SHADER, botVS);
    gBotDepthProg = LinkProgram(bvs, bfs);
    gBotDepth_uLightVP = glGetUniformLocation(gBotDepthProg, "uLightVP");
    gBotDepth_uModel = glGetUniformLocation(gBotDepthProg, "uModel");
    gBotDepth_uJoints = glGetUniformLocation(gBotDepthProg, "jointMatrices");
}

static void updateCamera(float dt) {
    glm::vec3 forward(
        cosf(pitch) * cosf(yaw),
        sinf(pitch),
        cosf(pitch) * sinf(yaw)
    );
    forward = glm::normalize(forward);
    glm::vec3 right = glm::normalize(glm::cross(forward, up));

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) eye_center += forward * camSpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) eye_center -= forward * camSpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) eye_center -= right * camSpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) eye_center += right * camSpeed * dt;

    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) eye_center.y += camSpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) eye_center.y -= camSpeed * dt;

    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)  yaw -= turnSpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) yaw += turnSpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)    pitch += turnSpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)  pitch -= turnSpeed * dt;

    pitch = glm::clamp(pitch, -1.2f, 1.2f);
    lookat = eye_center + forward;
}

static glm::vec3 lightIntensity(5e6f, 5e6f, 5e6f);
static glm::vec3 lightPosition(-275.0f, 500.0f, 800.0f);

static bool playAnimation = true;
static float playbackSpeed = 2.0f;

static const char* SKY_PX_PATH = "C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/skybox/right.png";
static const char* SKY_NX_PATH = "C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/skybox/left.png";
static const char* SKY_PY_PATH = "C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/skybox/top.png";
static const char* SKY_NY_PATH = "C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/skybox/bottom.png";
static const char* SKY_PZ_PATH = "C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/skybox/front.png";
static const char* SKY_NZ_PATH = "C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/skybox/back.png";

static const char* BOT_GLTF_PATH = "C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/model/bot/bot.gltf";
static const char* BOT_VERT_PATH = "C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/shader/bot.vert";
static const char* BOT_FRAG_PATH = "C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/shader/bot.frag";

static const char* SKYBOX_VERT_PATH =
"C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/shader/skybox.vert";
static const char* SKYBOX_FRAG_PATH =
"C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/shader/skybox.frag";

static const char* CLOUD_GLTF_PATH = "C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/cloud/scene.gltf";
static const char* CLOUD_VERT_PATH = "C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/shader/cloud.vert";
static const char* CLOUD_FRAG_PATH = "C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/shader/cloud.frag";
static const char* CLOUD_COLOR_PATH = "C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/cloud/textures/Cloud_baseColor.png";
static const char* CLOUD_NORMAL_PATH = "C:/Users/avamc/OneDrive/Desktop/College/Year4/graphics/final_project/final_project/cloud/textures/Cloud_normal.png";

static const float BOT_SCALE = 1.5f;
static const float CLOUD_SCALE = 45.0f;     
static const float CLOUD_Y = 200.0f;     
static const float CLOUD_SPACING = 1400.0f;   
static const float CLOUD_SCALE_JITTER = 0.35f; 
static const float CLOUD_LAYER_LOW = 160.0f;
static const float CLOUD_LAYER_HIGH = 330.0f;
static const float CLOUD_LAYER_BLEND = 70.0f; 
static const float CLOUD_Y_JITTER = 120.0f;   
static const float BOT_Y_OFFSET = 0.0f;    
static const int CLOUD_RADIUS = 5; 
static const float BOT_SPAWN_CHANCE = 0.7f;

static glm::mat4 computeLightVP() {
    glm::vec3 center = eye_center;
    glm::vec3 lightDir = glm::normalize(center - lightPosition);
    glm::vec3 lightPos = center - lightDir * 2000.0f;

    glm::mat4 lightView = glm::lookAt(lightPos, center, glm::vec3(0, 1, 0));

    float r = CLOUD_SPACING * (CLOUD_RADIUS + 1);
    float nearP = 0.1f;
    float farP = 7000.0f;

    glm::mat4 lightProj = glm::ortho(-r, r, -r, r, nearP, farP);
    return lightProj * lightView;
}

static GLuint LoadTexture2D(const char* path, bool flipY, bool wantAlpha) {
    int w, h, channels;
    stbi_set_flip_vertically_on_load(flipY ? 1 : 0);

    int req = wantAlpha ? 4 : 3;
    unsigned char* img = stbi_load(path, &w, &h, &channels, req);
    if (!img) {
        std::cerr << "Failed to load texture: " << path << "\n";
        return 0;
    }

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLenum fmt = wantAlpha ? GL_RGBA : GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, fmt, w, h, 0, fmt, GL_UNSIGNED_BYTE, img);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(img);
    return tex;
}

static GLuint LoadCubemap6(
    const char* px, const char* nx,
    const char* py, const char* ny,
    const char* pz, const char* nz,
    bool flipY
) {
    stbi_set_flip_vertically_on_load(flipY ? 1 : 0);

    const char* faces[6] = { px, nx, py, ny, pz, nz };
    GLenum targets[6] = {
        GL_TEXTURE_CUBE_MAP_POSITIVE_X,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
    };

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);

    int w = 0, h = 0, ch = 0;
    for (int i = 0; i < 6; ++i) {
        unsigned char* img = stbi_load(faces[i], &w, &h, &ch, 0);
        if (!img) {
            std::cerr << "Failed to load cubemap face: " << faces[i] << "\n";
            glDeleteTextures(1, &tex);
            return 0;
        }

        GLenum fmt = (ch == 4) ? GL_RGBA : GL_RGB;
        glTexImage2D(targets[i], 0, fmt, w, h, 0, fmt, GL_UNSIGNED_BYTE, img);
        stbi_image_free(img);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return tex;
}

// HASH CODE ASSISTED BY AI

static inline uint32_t hash2i(int x, int z) {
    uint32_t h = 2166136261u;
    h = (h ^ (uint32_t)x) * 16777619u;
    h = (h ^ (uint32_t)z) * 16777619u;
    return h;
}
static inline float hash01(uint32_t h) {
    return (h & 0x00FFFFFFu) / float(0x01000000u);
}

static inline float hashSigned01(uint32_t h) {
    return hash01(h) * 2.0f - 1.0f; 
}

struct Skybox {
    GLuint vao = 0, vboPos = 0, ebo = 0;
    GLuint program = 0;
    GLuint cubemap = 0;
    GLint vpLoc = -1;
    GLint cubeLoc = -1;

    float positions[24 * 3] = {
        // Front (+Z)
        -1,-1, 1,  
        1,-1, 1,  
        1, 1, 1,  
        -1, 1, 1,

        // Back (-Z)
         1,-1,-1, 
         -1,-1,-1, 
         -1, 1,-1,  
         1, 1,-1,

         // Left (-X)
         -1,-1,-1, 
         -1,-1, 1, 
         -1, 1, 1, 
         -1, 1,-1,

         // Right (+X)
          1,-1, 1,  
          1,-1,-1,  
          1, 1,-1,  
          1, 1, 1,

          // Top (+Y)
          -1, 1, 1,  
          1, 1, 1,  
          1, 1,-1, 
          -1, 1,-1,

          // Bottom (-Y)
          -1,-1,-1,  
          1,-1,-1,  
          1,-1, 1, 
          -1,-1, 1
    };

    unsigned int indices[36] = {
        0,1,2,  0,2,3,
        4,5,6,  4,6,7,
        8,9,10, 8,10,11,
        12,13,14, 12,14,15,
        16,17,18, 16,18,19,
        20,21,22, 20,22,23
    };

    void initialize() {

        program = LoadShadersFromFile(SKYBOX_VERT_PATH, SKYBOX_FRAG_PATH);
        if (program == 0) std::cerr << "Failed to load skybox shaders.\n";

        vpLoc = glGetUniformLocation(program, "uVP");
        cubeLoc = glGetUniformLocation(program, "uCube");

        cubemap = LoadCubemap6(
            SKY_PX_PATH, SKY_NX_PATH,
            SKY_PY_PATH, SKY_NY_PATH,
            SKY_PZ_PATH, SKY_NZ_PATH,
            false 
        );
        if (!cubemap) std::cerr << "Cubemap missing.\n";

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vboPos);
        glBindBuffer(GL_ARRAY_BUFFER, vboPos);
        glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        glBindVertexArray(0);
    }

    void render(const glm::mat4& projection, const glm::mat4& viewNoTranslation) {
        glDepthFunc(GL_LEQUAL);     
        glUseProgram(program);

        glm::mat4 vp = projection * viewNoTranslation;
        glUniformMatrix4fv(vpLoc, 1, GL_FALSE, glm::value_ptr(vp));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);
        glUniform1i(cubeLoc, 0);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (void*)0);
        glBindVertexArray(0);

        glDepthFunc(GL_LESS);
    }

    void cleanup() {
        if (program) glDeleteProgram(program);
        if (cubemap) glDeleteTextures(1, &cubemap);
        if (vboPos) glDeleteBuffers(1, &vboPos);
        if (ebo) glDeleteBuffers(1, &ebo);
        if (vao) glDeleteVertexArrays(1, &vao);
    }
};

struct Cloud {
    tinygltf::Model model;
    GLuint vao = 0, vboPos = 0, vboUV = 0, ebo = 0;
    GLuint program = 0;
    GLint mvpLoc = -1, colorLoc = -1;
    GLuint colorTex = 0;
    GLuint normalTex = 0;

    std::vector<float> positions;
    std::vector<float> uvs;
    std::vector<unsigned int> indices;

    GLuint vboN = 0;
    std::vector<float> normals;

    glm::vec3 localCenter = glm::vec3(0.0f);
    float localTopY = 0.0f;

    GLint modelLoc = -1;
    GLint camPosLoc = -1;
    GLint fogColorLoc = -1;
    GLint fogStartLoc = -1;
    GLint fogEndLoc = -1;

    bool loadGLTFMesh(const char* gltfPath) {
        tinygltf::TinyGLTF loader;
        std::string err, warn;
        bool ok = loader.LoadASCIIFromFile(&model, &err, &warn, gltfPath);
        if (!ok) {
            std::cerr << "Failed to load cloud gltf: " << gltfPath << "\n";
            return false;
        }

        const tinygltf::Primitive& prim = model.meshes[0].primitives[0];

        auto itPos = prim.attributes.find("POSITION");
        auto itUV = prim.attributes.find("TEXCOORD_0");

        const tinygltf::Accessor& posAcc = model.accessors[itPos->second];
        const tinygltf::BufferView& posView = model.bufferViews[posAcc.bufferView];
        const tinygltf::Buffer& posBuf = model.buffers[posView.buffer];
        const unsigned char* posPtr = posBuf.data.data() + posView.byteOffset + posAcc.byteOffset;
        positions.resize(posAcc.count * 3);
        memcpy(positions.data(), posPtr, positions.size() * sizeof(float));

        const tinygltf::Accessor& uvAcc = model.accessors[itUV->second];
        const tinygltf::BufferView& uvView = model.bufferViews[uvAcc.bufferView];
        const tinygltf::Buffer& uvBuf = model.buffers[uvView.buffer];
        const unsigned char* uvPtr = uvBuf.data.data() + uvView.byteOffset + uvAcc.byteOffset;
        uvs.resize(uvAcc.count * 2);
        memcpy(uvs.data(), uvPtr, uvs.size() * sizeof(float));

        auto itN = prim.attributes.find("NORMAL");
        if (itN == prim.attributes.end()) {
            std::cerr << "Cloud mesh missing NORMAL.\n";
            return false;
        }
        const tinygltf::Accessor& nAcc = model.accessors[itN->second];
        const tinygltf::BufferView& nView = model.bufferViews[nAcc.bufferView];
        const tinygltf::Buffer& nBuf = model.buffers[nView.buffer];
        const unsigned char* nPtr = nBuf.data.data() + nView.byteOffset + nAcc.byteOffset;

        normals.resize(nAcc.count * 3);
        memcpy(normals.data(), nPtr, normals.size() * sizeof(float));

        if (prim.indices < 0) {
            return false;
        }
        const tinygltf::Accessor& idxAcc = model.accessors[prim.indices];
        const tinygltf::BufferView& idxView = model.bufferViews[idxAcc.bufferView];
        const tinygltf::Buffer& idxBuf = model.buffers[idxView.buffer];
        const unsigned char* idxPtr = idxBuf.data.data() + idxView.byteOffset + idxAcc.byteOffset;

        indices.resize(idxAcc.count);
        if (idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
            const unsigned short* src = (const unsigned short*)idxPtr;
            for (size_t i = 0; i < indices.size(); ++i) indices[i] = (unsigned int)src[i];
        }
        else if (idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
            const unsigned int* src = (const unsigned int*)idxPtr;
            for (size_t i = 0; i < indices.size(); ++i) indices[i] = src[i];
        }
        else {
            return false;
        }

        glm::vec3 mn(1e30f), mx(-1e30f);
        for (size_t i = 0; i + 2 < positions.size(); i += 3) {
            glm::vec3 p(positions[i + 0], positions[i + 1], positions[i + 2]);
            mn = glm::min(mn, p);
            mx = glm::max(mx, p);
        }
        localCenter = 0.5f * (mn + mx);
        localTopY = mx.y; 

        return true;
    }

    void initialize() {
        if (!loadGLTFMesh(CLOUD_GLTF_PATH)) return;

        colorTex = LoadTexture2D(CLOUD_COLOR_PATH, true, true);  
        normalTex = LoadTexture2D(CLOUD_NORMAL_PATH, true, false); 

        program = LoadShadersFromFile(CLOUD_VERT_PATH, CLOUD_FRAG_PATH);
        if (program == 0) std::cerr << "Failed to load cloud shaders.\n";

        mvpLoc = glGetUniformLocation(program, "uMVP");
        colorLoc = glGetUniformLocation(program, "ucolor");

        modelLoc = glGetUniformLocation(program, "uModel");
        camPosLoc = glGetUniformLocation(program, "cameraPosition");
        fogColorLoc = glGetUniformLocation(program, "fogColor");
        fogStartLoc = glGetUniformLocation(program, "fogStart");
        fogEndLoc = glGetUniformLocation(program, "fogEnd");

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vboPos);
        glBindBuffer(GL_ARRAY_BUFFER, vboPos);
        glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float), positions.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glGenBuffers(1, &vboUV);
        glBindBuffer(GL_ARRAY_BUFFER, vboUV);
        glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(float), uvs.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glGenBuffers(1, &vboN);
        glBindBuffer(GL_ARRAY_BUFFER, vboN);
        glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

        glBindVertexArray(0);
    }

    void render(const glm::mat4& vp, const glm::mat4& modelMat) {
        if (!program || !vao || !colorTex) return;

        glUseProgram(program);
        glm::mat4 mvp = vp * modelMat;
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colorTex);
        glUniform1i(colorLoc, 0);

        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_2D, gShadowTex);
        glUniform1i(glGetUniformLocation(program, "uShadowMap"), 7);

        glUniformMatrix4fv(glGetUniformLocation(program, "uLightVP"), 1, GL_FALSE, glm::value_ptr(gLightVP));

        glUniform3fv(glGetUniformLocation(program, "lightPosition"), 1, &lightPosition[0]);
        glUniform3fv(glGetUniformLocation(program, "lightIntensity"), 1, &lightIntensity[0]);

        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(modelMat));

        glUniform3fv(camPosLoc, 1, &eye_center[0]);

        glm::vec3 fogCol(0.6f, 0.7f, 0.85f);
        glUniform3fv(fogColorLoc, 1, &fogCol[0]);

        glUniform1f(fogStartLoc, 1200.0f);
        glUniform1f(fogEndLoc, 6000.0f);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_CULL_FACE);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, (void*)0);
        glBindVertexArray(0);

        glDisable(GL_BLEND);
        glEnable(GL_CULL_FACE);
    }

    void cleanup() {
        if (program) glDeleteProgram(program);
        if (colorTex) glDeleteTextures(1, &colorTex);
        if (normalTex) glDeleteTextures(1, &normalTex);
        if (vboPos) glDeleteBuffers(1, &vboPos);
        if (vboUV) glDeleteBuffers(1, &vboUV);
        if (ebo) glDeleteBuffers(1, &ebo);
        if (vao) glDeleteVertexArrays(1, &vao);
    }
};

struct MyBot {
    GLuint mvpMatrixID = 0;
    GLuint jointMatricesID = 0;
    GLuint lightPositionID = 0;
    GLuint lightIntensityID = 0;
    GLuint programID = 0;

    GLint cameraPosID = -1;
    GLint fogColorID = -1;
    GLint fogStartID = -1;
    GLint fogEndID = -1;

    tinygltf::Model model;

    GLint modelID = -1;

    struct PrimitiveObject {
        GLuint vao;
        std::map<int, GLuint> vbos;
    };
    std::vector<PrimitiveObject> primitiveObjects;

    struct SkinObject {
        std::vector<glm::mat4> inverseBindMatrices;
        std::vector<glm::mat4> globalJointTransforms;
        std::vector<glm::mat4> jointMatrices;
    };
    std::vector<SkinObject> skinObjects;

    struct SamplerObject {
        std::vector<float> input;
        std::vector<glm::vec4> output;
        int interpolation;
    };
    struct AnimationObject { std::vector<SamplerObject> samplers; };
    std::vector<AnimationObject> animationObjects;

    glm::mat4 getNodeTransform(const tinygltf::Node& node) {
        glm::mat4 transform(1.0f);
        if (node.matrix.size() == 16) {
            transform = glm::make_mat4(node.matrix.data());
        }
        else {
            if (node.translation.size() == 3)
                transform = glm::translate(transform, glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
            if (node.rotation.size() == 4) {
                glm::quat q(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
                transform *= glm::mat4_cast(q);
            }
            if (node.scale.size() == 3)
                transform = glm::scale(transform, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
        }
        return transform;
    }

    void computeLocalNodeTransform(const tinygltf::Model& model, int nodeIndex, std::vector<glm::mat4>& localTransforms) {
        const tinygltf::Node& node = model.nodes[nodeIndex];
        localTransforms[nodeIndex] = getNodeTransform(node);
        for (size_t i = 0; i < node.children.size(); ++i)
            computeLocalNodeTransform(model, node.children[i], localTransforms);
    }

    void computeGlobalNodeTransform(const tinygltf::Model& model,
        const std::vector<glm::mat4>& localTransforms,
        int nodeIndex, const glm::mat4& parentTransform,
        std::vector<glm::mat4>& globalTransforms) {
        glm::mat4 global = parentTransform * localTransforms[nodeIndex];
        globalTransforms[nodeIndex] = global;
        const tinygltf::Node& node = model.nodes[nodeIndex];
        for (size_t i = 0; i < node.children.size(); ++i)
            computeGlobalNodeTransform(model, localTransforms, node.children[i], global, globalTransforms);
    }

    std::vector<SkinObject> prepareSkinning(const tinygltf::Model& model) {
        std::vector<SkinObject> out;
        for (size_t i = 0; i < model.skins.size(); i++) {
            SkinObject skinObject;
            const tinygltf::Skin& skin = model.skins[i];

            const tinygltf::Accessor& accessor = model.accessors[skin.inverseBindMatrices];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            const float* ptr = reinterpret_cast<const float*>(buffer.data.data() + accessor.byteOffset + bufferView.byteOffset);

            skinObject.inverseBindMatrices.resize(accessor.count);
            for (size_t j = 0; j < accessor.count; j++) {
                float m[16];
                memcpy(m, ptr + j * 16, 16 * sizeof(float));
                skinObject.inverseBindMatrices[j] = glm::make_mat4(m);
            }

            skinObject.globalJointTransforms.resize(skin.joints.size());
            skinObject.jointMatrices.resize(skin.joints.size());

            std::vector<glm::mat4> localTransforms(model.nodes.size(), glm::mat4(1.0f));
            std::vector<glm::mat4> globalTransforms(model.nodes.size(), glm::mat4(1.0f));
            int rootNodeIndex = (skin.skeleton >= 0) ? skin.skeleton : skin.joints[0];

            computeLocalNodeTransform(model, rootNodeIndex, localTransforms);
            computeGlobalNodeTransform(model, localTransforms, rootNodeIndex, glm::mat4(1.0f), globalTransforms);

            for (size_t j = 0; j < skin.joints.size(); ++j) {
                int jointNodeIndex = skin.joints[j];
                skinObject.globalJointTransforms[j] = globalTransforms[jointNodeIndex];
                skinObject.jointMatrices[j] = skinObject.globalJointTransforms[j] * skinObject.inverseBindMatrices[j];
            }

            out.push_back(skinObject);
        }
        return out;
    }

    int findKeyframeIndex(const std::vector<float>& times, float animationTime) {
        int left = 0;
        int right = (int)times.size() - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (mid + 1 < (int)times.size() && times[mid] <= animationTime && animationTime < times[mid + 1]) return mid;
            else if (times[mid] > animationTime) right = mid - 1;
            else left = mid + 1;
        }
        return (int)times.size() - 2;
    }

    std::vector<AnimationObject> prepareAnimation(const tinygltf::Model& model) {
        std::vector<AnimationObject> animationObjects;
        for (const auto& anim : model.animations) {
            AnimationObject animationObject;
            for (const auto& sampler : anim.samplers) {
                SamplerObject samplerObject;

                const tinygltf::Accessor& inputAccessor = model.accessors[sampler.input];
                const tinygltf::BufferView& inputBufferView = model.bufferViews[inputAccessor.bufferView];
                const tinygltf::Buffer& inputBuffer = model.buffers[inputBufferView.buffer];
                samplerObject.input.resize(inputAccessor.count);

                const unsigned char* inputPtr = &inputBuffer.data[inputBufferView.byteOffset + inputAccessor.byteOffset];
                int stride = inputAccessor.ByteStride(inputBufferView);
                for (size_t i = 0; i < inputAccessor.count; ++i)
                    samplerObject.input[i] = *reinterpret_cast<const float*>(inputPtr + i * stride);

                const tinygltf::Accessor& outputAccessor = model.accessors[sampler.output];
                const tinygltf::BufferView& outputBufferView = model.bufferViews[outputAccessor.bufferView];
                const tinygltf::Buffer& outputBuffer = model.buffers[outputBufferView.buffer];

                const unsigned char* outputPtr = &outputBuffer.data[outputBufferView.byteOffset + outputAccessor.byteOffset];
                samplerObject.output.resize(outputAccessor.count);

                for (size_t i = 0; i < outputAccessor.count; ++i) {
                    if (outputAccessor.type == TINYGLTF_TYPE_VEC3)
                        memcpy(&samplerObject.output[i], outputPtr + i * 3 * sizeof(float), 3 * sizeof(float));
                    else if (outputAccessor.type == TINYGLTF_TYPE_VEC4)
                        memcpy(&samplerObject.output[i], outputPtr + i * 4 * sizeof(float), 4 * sizeof(float));
                }

                animationObject.samplers.push_back(samplerObject);
            }
            animationObjects.push_back(animationObject);
        }
        return animationObjects;
    }

    void updateAnimation(const tinygltf::Model& model,
        const tinygltf::Animation& anim,
        const AnimationObject& animationObject,
        float time,
        std::vector<glm::mat4>& nodeTransforms) {
        for (const auto& channel : anim.channels) {
            int targetNodeIndex = channel.target_node;
            if (targetNodeIndex < 0 || (size_t)targetNodeIndex >= nodeTransforms.size()) continue;

            const auto& sampler = anim.samplers[channel.sampler];
            const tinygltf::Accessor& outputAccessor = model.accessors[sampler.output];
            const tinygltf::BufferView& outputBufferView = model.bufferViews[outputAccessor.bufferView];
            const tinygltf::Buffer& outputBuffer = model.buffers[outputBufferView.buffer];
            const std::vector<float>& times = animationObject.samplers[channel.sampler].input;
            if (times.size() < 2) continue;

            float animationTime = fmod(time, times.back());
            int keyframeIndex = findKeyframeIndex(times, animationTime);
            int nextIndex = glm::min(keyframeIndex + 1, (int)times.size() - 1);

            float t0 = times[keyframeIndex];
            float t1 = times[nextIndex];
            float factor = (t1 > t0) ? (animationTime - t0) / (t1 - t0) : 0.0f;
            factor = glm::clamp(factor, 0.0f, 1.0f);

            const unsigned char* outputPtr = &outputBuffer.data[outputBufferView.byteOffset + outputAccessor.byteOffset];

            glm::vec3 T;
            glm::vec3 S;
            glm::quat R;

            glm::mat4& M = nodeTransforms[targetNodeIndex];

            T = glm::vec3(M[3]);
            S.x = glm::length(glm::vec3(M[0]));
            S.y = glm::length(glm::vec3(M[1]));
            S.z = glm::length(glm::vec3(M[2]));

            glm::mat3 rotMat(
                glm::vec3(M[0]) / (S.x == 0 ? 1.f : S.x),
                glm::vec3(M[1]) / (S.y == 0 ? 1.f : S.y),
                glm::vec3(M[2]) / (S.z == 0 ? 1.f : S.z)
            );
            R = glm::quat_cast(rotMat);

            if (channel.target_path == "translation") {
                glm::vec3 v0, v1;
                memcpy(&v0, outputPtr + keyframeIndex * 3 * sizeof(float), 3 * sizeof(float));
                memcpy(&v1, outputPtr + nextIndex * 3 * sizeof(float), 3 * sizeof(float));
                T = glm::mix(v0, v1, factor);
            }
            else if (channel.target_path == "rotation") {
                glm::quat q0, q1;
                memcpy(&q0, outputPtr + keyframeIndex * 4 * sizeof(float), 4 * sizeof(float));
                memcpy(&q1, outputPtr + nextIndex * 4 * sizeof(float), 4 * sizeof(float));
                q0 = glm::normalize(q0);
                q1 = glm::normalize(q1);
                R = glm::normalize(glm::slerp(q0, q1, factor));
            }
            else if (channel.target_path == "scale") {
                glm::vec3 s0, s1;
                memcpy(&s0, outputPtr + keyframeIndex * 3 * sizeof(float), 3 * sizeof(float));
                memcpy(&s1, outputPtr + nextIndex * 3 * sizeof(float), 3 * sizeof(float));
                S = glm::mix(s0, s1, factor);
            }

            M = glm::translate(glm::mat4(1.0f), T) * glm::mat4_cast(R) * glm::scale(glm::mat4(1.0f), S);
        }
    }

    void updateSkinning(const std::vector<glm::mat4>& globalNodeTransforms) {
        for (size_t i = 0; i < model.skins.size(); ++i) {
            const tinygltf::Skin& skin = model.skins[i];
            SkinObject& skinObject = skinObjects[i];
            for (size_t j = 0; j < skin.joints.size(); ++j) {
                int jointNodeIndex = skin.joints[j];
                skinObject.globalJointTransforms[j] = globalNodeTransforms[jointNodeIndex];
                skinObject.jointMatrices[j] = skinObject.globalJointTransforms[j] * skinObject.inverseBindMatrices[j];
            }
        }
    }

    void update(float time) {
        if (model.skins.empty()) return;

        std::vector<glm::mat4> localTransforms(model.nodes.size(), glm::mat4(1.0f));
        const tinygltf::Skin& skin = model.skins[0];
        int rootIndex = (skin.skeleton >= 0) ? skin.skeleton : skin.joints[0];

        computeLocalNodeTransform(model, rootIndex, localTransforms);

        if (!model.animations.empty() && !animationObjects.empty()) {
            const tinygltf::Animation& anim = model.animations[0];
            const AnimationObject& animationObject = animationObjects[0];
            updateAnimation(model, anim, animationObject, time, localTransforms);
        }

        std::vector<glm::mat4> globalTransforms(model.nodes.size(), glm::mat4(1.0f));
        computeGlobalNodeTransform(model, localTransforms, rootIndex, glm::mat4(1.0f), globalTransforms);
        updateSkinning(globalTransforms);
    }

    bool loadModel(tinygltf::Model& model, const char* filename) {
        tinygltf::TinyGLTF loader;
        std::string err, warn;
        bool res = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
        if (!res) std::cout << "Failed to load glTF: " << filename << "\n";
        else std::cout << "Loaded glTF: " << filename << "\n";
        return res;
    }

    void bindMesh(std::vector<PrimitiveObject>& primitiveObjects, tinygltf::Model& model, tinygltf::Mesh& mesh) {
        std::map<int, GLuint> vbos;

        for (size_t i = 0; i < model.bufferViews.size(); ++i) {
            const tinygltf::BufferView& bufferView = model.bufferViews[i];
            if (bufferView.target == 0) continue;

            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            GLuint vbo;
            glGenBuffers(1, &vbo);
            glBindBuffer(bufferView.target, vbo);
            glBufferData(bufferView.target, bufferView.byteLength,
                &buffer.data.at(0) + bufferView.byteOffset, GL_STATIC_DRAW);
            vbos[(int)i] = vbo;
        }

        for (size_t i = 0; i < mesh.primitives.size(); ++i) {
            tinygltf::Primitive primitive = mesh.primitives[i];

            GLuint vao;
            glGenVertexArrays(1, &vao);
            glBindVertexArray(vao);

            for (auto& attrib : primitive.attributes) {
                tinygltf::Accessor accessor = model.accessors[attrib.second];
                int byteStride = accessor.ByteStride(model.bufferViews[accessor.bufferView]);

                glBindBuffer(GL_ARRAY_BUFFER, vbos[accessor.bufferView]);

                int size = (accessor.type == TINYGLTF_TYPE_SCALAR) ? 1 : accessor.type;

                int vaa = -1;
                if (attrib.first == "POSITION") vaa = 0;
                if (attrib.first == "NORMAL")   vaa = 1;
                if (attrib.first == "TEXCOORD_0") vaa = 2;
                if (attrib.first == "JOINTS_0") vaa = 3;
                if (attrib.first == "WEIGHTS_0") vaa = 4;

                if (vaa > -1) {
                    glEnableVertexAttribArray(vaa);
                    glVertexAttribPointer(vaa, size, accessor.componentType,
                        accessor.normalized ? GL_TRUE : GL_FALSE,
                        byteStride, BUFFER_OFFSET(accessor.byteOffset));
                }
            }

            PrimitiveObject po;
            po.vao = vao;
            po.vbos = vbos;
            primitiveObjects.push_back(po);

            glBindVertexArray(0);
        }
    }

    void bindModelNodes(std::vector<PrimitiveObject>& primitiveObjects, tinygltf::Model& model, tinygltf::Node& node) {
        if ((node.mesh >= 0) && (node.mesh < (int)model.meshes.size()))
            bindMesh(primitiveObjects, model, model.meshes[node.mesh]);
        for (size_t i = 0; i < node.children.size(); i++)
            bindModelNodes(primitiveObjects, model, model.nodes[node.children[i]]);
    }

    std::vector<PrimitiveObject> bindModel(tinygltf::Model& model) {
        std::vector<PrimitiveObject> primitiveObjects;
        const tinygltf::Scene& scene = model.scenes[model.defaultScene];
        for (size_t i = 0; i < scene.nodes.size(); ++i)
            bindModelNodes(primitiveObjects, model, model.nodes[scene.nodes[i]]);
        return primitiveObjects;
    }

    void drawMesh(const std::vector<PrimitiveObject>& primitiveObjects, tinygltf::Model& model, tinygltf::Mesh& mesh) {
        for (size_t i = 0; i < mesh.primitives.size(); ++i) {
            if (i >= primitiveObjects.size()) return;

            GLuint vao = primitiveObjects[i].vao;
            std::map<int, GLuint> vbos = primitiveObjects[i].vbos;

            glBindVertexArray(vao);

            tinygltf::Primitive primitive = mesh.primitives[i];
            tinygltf::Accessor indexAccessor = model.accessors[primitive.indices];
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbos.at(indexAccessor.bufferView));

            glDrawElements(primitive.mode, (GLsizei)indexAccessor.count,
                indexAccessor.componentType,
                BUFFER_OFFSET(indexAccessor.byteOffset));

            glBindVertexArray(0);
        }
    }

    void drawModelNodes(const std::vector<PrimitiveObject>& primitiveObjects, tinygltf::Model& model, tinygltf::Node& node) {
        if ((node.mesh >= 0) && (node.mesh < (int)model.meshes.size()))
            drawMesh(primitiveObjects, model, model.meshes[node.mesh]);
        for (size_t i = 0; i < node.children.size(); i++)
            drawModelNodes(primitiveObjects, model, model.nodes[node.children[i]]);
    }

    void drawModel(const std::vector<PrimitiveObject>& primitiveObjects, tinygltf::Model& model) {
        const tinygltf::Scene& scene = model.scenes[model.defaultScene];
        for (size_t i = 0; i < scene.nodes.size(); ++i)
            drawModelNodes(primitiveObjects, model, model.nodes[scene.nodes[i]]);
    }

    void initialize() {
        if (!loadModel(model, BOT_GLTF_PATH)) return;
        primitiveObjects = bindModel(model);
        skinObjects = prepareSkinning(model);
        animationObjects = prepareAnimation(model);

        programID = LoadShadersFromFile(BOT_VERT_PATH, BOT_FRAG_PATH);
        if (programID == 0) std::cerr << "Failed to load bot shaders.\n";

        modelID = glGetUniformLocation(programID, "uModel");

        cameraPosID = glGetUniformLocation(programID, "cameraPosition");
        fogColorID = glGetUniformLocation(programID, "fogColor");
        fogStartID = glGetUniformLocation(programID, "fogStart");
        fogEndID = glGetUniformLocation(programID, "fogEnd");

        mvpMatrixID = glGetUniformLocation(programID, "MVP");
        lightPositionID = glGetUniformLocation(programID, "lightPosition");
        lightIntensityID = glGetUniformLocation(programID, "lightIntensity");
        jointMatricesID = glGetUniformLocation(programID, "jointMatrices");
    }

    void render(const glm::mat4& vp, const glm::mat4& modelMatrix) {
        glUseProgram(programID);

        glm::mat4 mvp = vp * modelMatrix;
        glUniformMatrix4fv(mvpMatrixID, 1, GL_FALSE, glm::value_ptr(mvp));

        glUniformMatrix4fv(modelID, 1, GL_FALSE, glm::value_ptr(modelMatrix));

        glUniform3fv(cameraPosID, 1, &eye_center[0]);

        glm::vec3 fogCol(0.6f, 0.7f, 0.85f);
        glUniform3fv(fogColorID, 1, &fogCol[0]);

        glUniform1f(fogStartID, 1200.0f);
        glUniform1f(fogEndID, 6000.0f);

        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_2D, gShadowTex);
        glUniform1i(glGetUniformLocation(programID, "uShadowMap"), 7);

        glUniformMatrix4fv(glGetUniformLocation(programID, "uLightVP"), 1, GL_FALSE, glm::value_ptr(gLightVP));

        if (!skinObjects.empty() && (GLint)jointMatricesID >= 0) {
            const SkinObject& skin = skinObjects[0];
            if (!skin.jointMatrices.empty()) {
                glUniformMatrix4fv(jointMatricesID, (GLsizei)skin.jointMatrices.size(), GL_FALSE,
                    glm::value_ptr(skin.jointMatrices[0]));
            }
        }

        glUniform3fv(lightPositionID, 1, &lightPosition[0]);
        glUniform3fv(lightIntensityID, 1, &lightIntensity[0]);

        drawModel(primitiveObjects, model);
    }

    void cleanup() { if (programID) glDeleteProgram(programID); }
};

static void renderCloudField(const glm::mat4& vp, Cloud& cloud, MyBot& bot, float t) {
    int baseX = (int)floorf(eye_center.x / CLOUD_SPACING);
    int baseZ = (int)floorf(eye_center.z / CLOUD_SPACING);

    for (int dz = -CLOUD_RADIUS; dz <= CLOUD_RADIUS; ++dz) {
        for (int dx = -CLOUD_RADIUS; dx <= CLOUD_RADIUS; ++dx) {
            int cx = baseX + dx;
            int cz = baseZ + dz;

            uint32_t h = hash2i(cx, cz);

            float jitterAmp = CLOUD_SPACING * 0.75f; 

            float jx = hashSigned01(h * 747796405u + 2891336453u) * jitterAmp;
            float jz = hashSigned01(h * 277803737u + 15485863u) * jitterAmp;

            float worldX = cx * CLOUD_SPACING + jx;
            float worldZ = cz * CLOUD_SPACING + jz;

            float layerPick = hash01(h * 9781u + 6271u);
            float baseLayer = (layerPick < 0.55f) ? CLOUD_LAYER_LOW : CLOUD_LAYER_HIGH;

            float yJitter = hashSigned01(h * 1597334677u + 3812015801u) * CLOUD_LAYER_BLEND;
            float cloudY = baseLayer + yJitter;

            float sJitter = hashSigned01(h * 2654435761u + 1013904223u) * CLOUD_SCALE_JITTER;
            float cloudScale = CLOUD_SCALE * (1.0f + sJitter);

            float rotY = hash01(h * 2246822519u + 3266489917u) * 6.2831853f;

            glm::mat4 cloudM =
                glm::translate(glm::mat4(1.0f), glm::vec3(worldX, cloudY, worldZ)) *
                glm::rotate(glm::mat4(1.0f), rotY, glm::vec3(0, 1, 0)) *
                glm::scale(glm::mat4(1.0f), glm::vec3(cloudScale));

            cloud.render(vp, cloudM);

            float r = hash01(h);
            if (r > BOT_SPAWN_CHANCE) continue;

            float phase = (h & 0xFFFFu) * (1.0f / 65535.0f) * 6.2831853f;
            float speed = 0.7f + 0.6f * hash01(h >> 8);

            glm::vec3 centerLocal = cloud.localCenter;
            glm::vec3 centerOffset = glm::vec3(
                centerLocal.x * cloudScale,
                centerLocal.y * cloudScale,
                centerLocal.z * cloudScale
            );

            glm::vec3 centerOffsetRot = glm::vec3(
                cosf(rotY) * centerOffset.x + sinf(rotY) * centerOffset.z,
                centerOffset.y,
                -sinf(rotY) * centerOffset.x + cosf(rotY) * centerOffset.z
            );

            glm::vec3 cloudCenterWorld = glm::vec3(worldX, cloudY, worldZ) + centerOffsetRot;

            float runRadius = 2.0f; 
            float ang = t * speed + phase;

            float bx = cloudCenterWorld.x + cosf(ang) * runRadius;
            float bz = cloudCenterWorld.z + sinf(ang) * runRadius;

            float by = cloudCenterWorld.y * 0.75f;

            float heading = ang + 1.5707963f;

            glm::mat4 botM =
                glm::translate(glm::mat4(1.0f), glm::vec3(bx, by, bz)) *
                glm::rotate(glm::mat4(1.0f), heading, glm::vec3(0, 1, 0)) *
                glm::scale(glm::mat4(1.0f), glm::vec3(BOT_SCALE));

            bot.render(vp, botM);
        }
    }
}

static void renderCloudFieldDepth(Cloud& cloud, MyBot& bot, float t) {
    int baseX = (int)floorf(eye_center.x / CLOUD_SPACING);
    int baseZ = (int)floorf(eye_center.z / CLOUD_SPACING);

    for (int dz = -CLOUD_RADIUS; dz <= CLOUD_RADIUS; ++dz) {
        for (int dx = -CLOUD_RADIUS; dx <= CLOUD_RADIUS; ++dx) {
            int cx = baseX + dx;
            int cz = baseZ + dz;

            uint32_t h = hash2i(cx, cz);

            float jitterAmp = CLOUD_SPACING * 0.75f;
            float jx = hashSigned01(h * 747796405u + 2891336453u) * jitterAmp;
            float jz = hashSigned01(h * 277803737u + 15485863u) * jitterAmp;

            float worldX = cx * CLOUD_SPACING + jx;
            float worldZ = cz * CLOUD_SPACING + jz;

            float layerPick = hash01(h * 9781u + 6271u);
            float baseLayer = (layerPick < 0.55f) ? CLOUD_LAYER_LOW : CLOUD_LAYER_HIGH;

            float yJitter = hashSigned01(h * 1597334677u + 3812015801u) * CLOUD_LAYER_BLEND;
            float cloudY = baseLayer + yJitter;

            float sJitter = hashSigned01(h * 2654435761u + 1013904223u) * CLOUD_SCALE_JITTER;
            float cloudScale = CLOUD_SCALE * (1.0f + sJitter);

            float rotY = hash01(h * 2246822519u + 3266489917u) * 6.2831853f;

            glm::mat4 cloudM =
                glm::translate(glm::mat4(1.0f), glm::vec3(worldX, cloudY, worldZ)) *
                glm::rotate(glm::mat4(1.0f), rotY, glm::vec3(0, 1, 0)) *
                glm::scale(glm::mat4(1.0f), glm::vec3(cloudScale));

            glUseProgram(gCloudDepthProg);
            glUniformMatrix4fv(gCloudDepth_uLightVP, 1, GL_FALSE, glm::value_ptr(gLightVP));
            glUniformMatrix4fv(gCloudDepth_uModel, 1, GL_FALSE, glm::value_ptr(cloudM));
            glBindVertexArray(cloud.vao);
            glDrawElements(GL_TRIANGLES, (GLsizei)cloud.indices.size(), GL_UNSIGNED_INT, (void*)0);
            glBindVertexArray(0);

            float r = hash01(h);
            if (r > BOT_SPAWN_CHANCE) continue;

            float phase = (h & 0xFFFFu) * (1.0f / 65535.0f) * 6.2831853f;
            float speed = 0.7f + 0.6f * hash01(h >> 8);

            glm::vec3 centerLocal = cloud.localCenter;
            glm::vec3 centerOffset = glm::vec3(centerLocal.x * cloudScale,
                centerLocal.y * cloudScale,
                centerLocal.z * cloudScale);

            glm::vec3 centerOffsetRot = glm::vec3(
                cosf(rotY) * centerOffset.x + sinf(rotY) * centerOffset.z,
                centerOffset.y,
                -sinf(rotY) * centerOffset.x + cosf(rotY) * centerOffset.z
            );

            glm::vec3 cloudCenterWorld = glm::vec3(worldX, cloudY, worldZ) + centerOffsetRot;

            float runRadius = 2.0f;
            float ang = t * speed + phase;

            float bx = cloudCenterWorld.x + cosf(ang) * runRadius;
            float bz = cloudCenterWorld.z + sinf(ang) * runRadius;
            float by = cloudCenterWorld.y * 0.75f;

            float heading = ang + 1.5707963f;

            glm::mat4 botM =
                glm::translate(glm::mat4(1.0f), glm::vec3(bx, by, bz)) *
                glm::rotate(glm::mat4(1.0f), heading, glm::vec3(0, 1, 0)) *
                glm::scale(glm::mat4(1.0f), glm::vec3(BOT_SCALE));

            glUseProgram(gBotDepthProg);
            glUniformMatrix4fv(gBotDepth_uLightVP, 1, GL_FALSE, glm::value_ptr(gLightVP));
            glUniformMatrix4fv(gBotDepth_uModel, 1, GL_FALSE, glm::value_ptr(botM));

            if (!bot.skinObjects.empty() && gBotDepth_uJoints >= 0) {
                const auto& skin = bot.skinObjects[0];
                if (!skin.jointMatrices.empty()) {
                    glUniformMatrix4fv(gBotDepth_uJoints, (GLsizei)skin.jointMatrices.size(),
                        GL_FALSE, glm::value_ptr(skin.jointMatrices[0]));
                }
            }

            bot.drawModel(bot.primitiveObjects, bot.model);
        }
    }
}

int main(void) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW.\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(windowWidth, windowHeight, "Final Project > FPS: ", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to open a GLFW window.\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    glfwSetKeyCallback(window, key_callback);

    int version = gladLoadGL(glfwGetProcAddress);
    if (version == 0) {
        return -1;
    }

    glClearColor(0.2f, 0.2f, 0.25f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    Skybox sky;
    sky.initialize();

    Cloud cloud;
    cloud.initialize();

    MyBot bot;
    bot.initialize();

    initShadowMap();
    initDepthPrograms();

    glm::mat4 projectionMatrix =
        glm::perspective(glm::radians(FoV), (float)windowWidth / (float)windowHeight, zNear, zFar);

    double lastTime = glfwGetTime();
    float time = 0.0f;
    float fTime = 0.0f;
    unsigned long frames = 0;

    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        float deltaTime = float(currentTime - lastTime);
        lastTime = currentTime;

        updateCamera(deltaTime);

        if (playAnimation) {
            time += deltaTime * playbackSpeed;
            bot.update(time);
        }

        glm::mat4 viewMatrix = glm::lookAt(eye_center, lookat, up);
        glm::mat4 vp = projectionMatrix * viewMatrix;

        gLightVP = computeLightVP();
        glViewport(0, 0, SHADOW_RES, SHADOW_RES);
        glBindFramebuffer(GL_FRAMEBUFFER, gShadowFBO);
        glClear(GL_DEPTH_BUFFER_BIT);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(2.0f, 4.0f);
        renderCloudFieldDepth(cloud, bot, (float)glfwGetTime());
        glDisable(GL_POLYGON_OFFSET_FILL);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, windowWidth, windowHeight);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDepthMask(GL_FALSE);
        glCullFace(GL_FRONT);
        glm::mat4 viewNoTrans = glm::mat4(glm::mat3(viewMatrix));
        sky.render(projectionMatrix, viewNoTrans);
        glCullFace(GL_BACK);
        glDepthMask(GL_TRUE);

        renderCloudField(vp, cloud, bot, (float)glfwGetTime());

        frames++;
        fTime += deltaTime;
        if (fTime > 2.0f) {
            float fps = frames / fTime;
            frames = 0;
            fTime = 0;
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2)
                << "Final Project > FPS: " << fps;
            glfwSetWindowTitle(window, stream.str().c_str());
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    bot.cleanup();
    cloud.cleanup();
    sky.cleanup();
    glfwTerminate();
    return 0;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key == GLFW_KEY_UP && action == GLFW_PRESS) {
        playbackSpeed += 1.0f;
        if (playbackSpeed > 10.0f) playbackSpeed = 10.0f;
    }
    if (key == GLFW_KEY_DOWN && action == GLFW_PRESS) {
        playbackSpeed -= 1.0f;
        if (playbackSpeed < 1.0f) playbackSpeed = 1.0f;
    }
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        playAnimation = !playAnimation;
    }
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}